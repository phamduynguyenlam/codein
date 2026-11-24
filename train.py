import os
import argparse
import copy
import gc
import math
import random
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.codein import DiscAF, CodeIn, tensor_policy_state
from solver.moead_ego import run_surrogate_moead_ego
from problem.constraints import evaluate_objectives_and_constraints, initial_constraint_scale
from problem.problem import DAS_PROBLEMS, MW_PROBLEMS, default_problem_dim, get_reference_point, make_problem, reference_front
from lhs import latin_hypercube_sample as _scipy_latin_hypercube_sample
from reward import (
    hypervolume,
    pareto_front,
    resolve_default_reward_lambda,
    reward_scheme_1,
    reward_scheme_2,
)
from surrogate.surrogate_model import (
    fit_gp_surrogates,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def dbg(msg):
    print(f"[DBG {time.strftime('%H:%M:%S')}] pid={os.getpid()} | {msg}", flush=True)


@dataclass
class TrainConfig:
    seed: int = 0
    num_workers: int = 24
    episodes_per_worker: int = 1
    max_fe: int = 300
    init_size: int = 100
    eval_batch_size: int = 5
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 0.99
    lr: float = 1e-4
    target_soft_tau: float = 1e-3
    train_iters: int = 50
    updates_per_epoch: int = 80
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_iters: int = 10
    hidden_dim: int = 128
    n_heads: int = 8
    ff_dim: int = 256
    dropout: float = 0.0
    logit_scale: float = 1.0
    surrogate_model: str = "gp"
    solver: str = "moead_ego"
    saea_steps: int = 15
    offspring_size: int = 40
    reward_scheme: int = 1
    reward_lambda: float = 5.0
    reward_norm: bool = True
    policy_mode: str = "epsilon_greedy"
    heldout_problem: str = "MW"
    weight_dir: str = "weight"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rollout_device: str = "cpu"
    surrogate_device: str = "cpu"
    agent_name: str = "codein"
    num_seeds: int = 8
    temperature: float = 1.0
    value_coef: float = 0.5
    q_coef: float = 1.0
    prediction_history_steps: int = 40
    cuda_cleanup_before_update: bool = False
    cuda_cleanup_after_update: bool = False


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def extend(self, transitions):
        self.buffer.extend(transitions)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


def epsilon_by_iter(it, cfg):
    frac = min(1.0, it / cfg.epsilon_decay_iters)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def seed_everything(seed: int, *, include_cuda: bool = True) -> None:
    resolved_seed = int(seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed % (2**32))
    torch.manual_seed(resolved_seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _estimate_transition_nbytes(transition) -> int:
    total = 0
    for item in transition:
        if isinstance(item, np.ndarray):
            total += int(item.nbytes)
    return int(total)


def _debug_enabled(cfg_like) -> bool:
    if cfg_like is None:
        return False
    if isinstance(cfg_like, dict):
        return bool(cfg_like.get("debug", False))
    return bool(getattr(cfg_like, "debug", False))


def _process_memory_stats_mb() -> dict[str, float]:
    stats = {
        "rss_mb": float("nan"),
        "vms_mb": float("nan"),
        "uss_mb": float("nan"),
        "avail_mb": float("nan"),
    }
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        stats["rss_mb"] = float(mem_info.rss) / (1024.0 * 1024.0)
        stats["vms_mb"] = float(mem_info.vms) / (1024.0 * 1024.0)
        try:
            full_info = proc.memory_full_info()
            if hasattr(full_info, "uss"):
                stats["uss_mb"] = float(full_info.uss) / (1024.0 * 1024.0)
        except Exception:
            pass
        try:
            stats["avail_mb"] = float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
        except Exception:
            pass
    except Exception:
        pass
    return stats


def _cuda_memory_stats_mb(device_like: str) -> dict[str, float]:
    stats = {
        "alloc_mb": 0.0,
        "reserved_mb": 0.0,
        "peak_alloc_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "free_mb": float("nan"),
        "total_mb": float("nan"),
    }
    if not torch.cuda.is_available():
        return stats
    try:
        device = torch.device(str(device_like))
        if device.type != "cuda":
            return stats
        device_index = torch.cuda.current_device() if device.index is None else int(device.index)
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        stats["alloc_mb"] = float(torch.cuda.memory_allocated(device_index)) / (1024.0 * 1024.0)
        stats["reserved_mb"] = float(torch.cuda.memory_reserved(device_index)) / (1024.0 * 1024.0)
        stats["peak_alloc_mb"] = float(torch.cuda.max_memory_allocated(device_index)) / (1024.0 * 1024.0)
        stats["peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(device_index)) / (1024.0 * 1024.0)
        stats["free_mb"] = float(free_b) / (1024.0 * 1024.0)
        stats["total_mb"] = float(total_b) / (1024.0 * 1024.0)
    except Exception:
        pass
    return stats


def _cleanup_cuda_cache(device_like: str, rounds: int = 3, sleep_sec: float = 1.0) -> dict[str, dict[str, float]]:
    before = _cuda_memory_stats_mb(device_like)
    if not torch.cuda.is_available():
        return {"before": before, "after": before}
    try:
        device = torch.device(str(device_like))
        if device.type != "cuda":
            return {"before": before, "after": before}
    except Exception:
        return {"before": before, "after": before}

    for _ in range(max(1, int(rounds))):
        gc.collect()
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if float(sleep_sec) > 0:
            time.sleep(float(sleep_sec))
    gc.collect()
    after = _cuda_memory_stats_mb(device_like)
    return {"before": before, "after": after}


def _nvidia_smi_snapshot() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout or "").strip()
        if output:
            return output.replace("\n", " ; ")
        if (result.stderr or "").strip():
            return f"stderr: {result.stderr.strip()}"
        return "no compute processes"
    except Exception as exc:
        return f"unavailable: {exc}"


def clone_state_dict_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def soft_update_target_network(target_model, online_model, tau: float) -> None:
    tau_value = float(tau)
    one_minus_tau = 1.0 - tau_value
    with torch.no_grad():
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.mul_(one_minus_tau).add_(online_param.data, alpha=tau_value)
        for target_buffer, online_buffer in zip(target_model.buffers(), online_model.buffers()):
            target_buffer.copy_(online_buffer)


def resolve_agent_cls(agent_name):
    name = str(agent_name).strip().lower()
    if name == "codein":
        return CodeIn
    if name == "disc_af":
        return DiscAF
    raise ValueError(f"Unsupported agent_name: {agent_name}")


def build_agent_init_kwargs(cfg_like, *, epsilon=None):
    get_value = cfg_like.get if isinstance(cfg_like, dict) else lambda key, default=None: getattr(cfg_like, key, default)
    kwargs = {
        "hidden_dim": int(get_value("hidden_dim")),
        "n_heads": int(get_value("n_heads")),
        "ff_dim": int(get_value("ff_dim")),
        "dropout": float(get_value("dropout")),
    }
    if get_value("agent_name", "codein") == "codein":
        return kwargs
    kwargs["logit_scale"] = float(get_value("logit_scale"))
    if epsilon is not None:
        kwargs["epsilon"] = float(epsilon)
    return kwargs


def select_action_from_output(out):
    if "action" in out:
        return int(out["action"].reshape(-1)[0].item())
    ranking = out["ranking"]
    return int(ranking[0, 0].item())


def parse_args(configure_parser=None):
    parser = argparse.ArgumentParser(description="Train CodeIn with surrogate-assisted environments.")
    parser.add_argument("--problem", type=str.upper, choices=("MW", "DAS"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--init_fe", type=int, default=TrainConfig.init_size)
    parser.add_argument("--max_fe", type=int, default=TrainConfig.max_fe)
    parser.add_argument("--batch", type=int, default=TrainConfig.eval_batch_size, help="Evaluations per candidate pool.")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--reward_scheme", type=int, default=1, choices=[1, 2])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_seeds", type=int, default=TrainConfig.num_seeds)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--value_coef", type=float, default=TrainConfig.value_coef)
    parser.add_argument("--q_coef", type=float, default=TrainConfig.q_coef)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()
    args.gamma = 0.99
    args.reward_norm = True
    args.surrogate_model = "gp"
    args.solver = "moead_ego"
    args.updates_per_epoch = None
    args.hidden_dim = None
    args.rollout_device = "cpu"
    args.surrogate_device = "cpu"
    args.agent_name = "codein"
    args.cuda_cleanup_before_update = None
    args.cuda_cleanup_after_update = None
    args.ray = False
    return args

def latin_hypercube_sample(n_samples, dim, lower, upper, seed):
    return _scipy_latin_hypercube_sample(
        n_samples=int(n_samples), dim=int(dim), lower=lower, upper=upper, seed=int(seed)
    )


def build_named_surrogate_from_cfg(cfg_dict, archive_x, archive_y, surrogate_name, existing_surrogate=None):
    del existing_surrogate
    surrogate_name = str(surrogate_name).lower()
    if surrogate_name == "gp":
        return fit_gp_surrogates(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            seed=int(cfg_dict.get("seed", 0)),
            nu=float(cfg_dict.get("gp_nu", 5.0)),
        )

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def build_surrogate_from_cfg(cfg_dict, archive_x, archive_y, existing_surrogate=None):
    surrogate_name = str(cfg_dict.get("surrogate_model", "gp")).lower()
    return build_named_surrogate_from_cfg(
        cfg_dict,
        archive_x,
        archive_y,
        surrogate_name,
        existing_surrogate=existing_surrogate,
    )


def make_nsga2_problem_adapter(problem, n_obj):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def generate_offspring_pool(cfg_dict, archive_x, archive_y, nsga2_problem, seed, seed_x=None):
    surrogate = build_named_surrogate_from_cfg(
        cfg_dict,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate_name="gp",
    )
    result = run_surrogate_moead_ego(
        problem=nsga2_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate=surrogate,
        pop_size=int(cfg_dict.get("offspring_size", TrainConfig.offspring_size)),
        saea_steps=int(cfg_dict.get("saea_steps", TrainConfig.saea_steps)),
        seed=int(seed),
        seed_x=seed_x,
    )
    return (
        surrogate,
        np.asarray(result.x, dtype=np.float32),
        np.asarray(result.mean, dtype=np.float32),
        np.asarray(result.std, dtype=np.float32),
        {},
    )


def pad_stack_rows(arrays, pad_value=0.0):
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty.")

    max_ndim = max(arr.ndim for arr in arrays)
    normalized = []
    for arr in arrays:
        if arr.ndim < max_ndim:
            new_shape = arr.shape + (1,) * (max_ndim - arr.ndim)
            arr = arr.reshape(new_shape)
        normalized.append(arr)

    target_shape = tuple(
        max(int(arr.shape[axis]) for arr in normalized)
        for axis in range(max_ndim)
    )

    padded = []
    for arr in normalized:
        if arr.shape == target_shape:
            padded.append(arr)
            continue

        out = np.full(target_shape, pad_value, dtype=np.float32)
        slices = tuple(slice(0, int(size)) for size in arr.shape)
        out[slices] = arr
        padded.append(out)

    return np.stack(padded, axis=0)


def build_row_mask(arrays):
    arrays = [np.asarray(arr) for arr in arrays]
    max_rows = max(int(arr.shape[0]) for arr in arrays)
    mask = np.zeros((len(arrays), max_rows), dtype=bool)
    for i, arr in enumerate(arrays):
        mask[i, : int(arr.shape[0])] = True
    return mask


def prediction_history_kwargs(agent, y_true_history, sigma_true_history):
    if not bool(getattr(agent, "uses_prediction_history", False)):
        return {}
    return {
        "y_true_history": y_true_history,
        "sigma_true_history": sigma_true_history,
    }


def _batch_codein_constraints(states, device, *, include_offspring):
    keys = ["g_true", "constraint_scale", "seed_count"]
    if include_offspring:
        keys.extend(("g_sur_mean", "g_sur_std"))
    out = {
        key: to_tensor(pad_stack_rows([state[key] for state in states]), device)
        for key in keys
    }
    out["constraint_mask"] = torch.as_tensor(
        build_row_mask([state["constraint_scale"] for state in states]), device=device
    )
    return out


def compute_codein_init_losses(agent, decisions, device):
    states, grades, _, returns = zip(*decisions)
    keys = ("x_true", "y_true", "progress", "lower_bound", "upper_bound")
    inputs = {key: to_tensor(pad_stack_rows([state[key] for state in states]), device) for key in keys}
    inputs.update(_batch_codein_constraints(states, device, include_offspring=False))
    inputs["archive_mask"] = torch.as_tensor(build_row_mask([state["x_true"] for state in states]), device=device)
    logits, values, mask = agent.init_policy(**inputs)
    actions = torch.as_tensor(pad_stack_rows(grades, pad_value=1), dtype=torch.long, device=device) - 1
    log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
    log_probs = log_probs.masked_fill(~mask, 0.0).sum(dim=1)
    returns = to_tensor(returns, device)
    advantage = (returns - values).detach()
    return -(advantage * log_probs).mean(), 0.5 * (values - returns.detach()).square().mean()


def backward_codein_init(agent, decisions, cfg):
    groups = {}
    for decision in decisions:
        groups.setdefault(decision[0]["y_true"].shape[1], []).append(decision)
    policy_loss, value_loss = 0.0, 0.0
    # Accumulate gradients in small groups instead of retaining every archive attention graph.
    for group in groups.values():
        for start in range(0, len(group), 4):
            chunk = group[start:start + 4]
            init, value = compute_codein_init_losses(agent, chunk, cfg.device)
            weight = len(chunk) / len(decisions)
            (weight * (init + cfg.value_coef * value)).backward()
            policy_loss += weight * init.detach().item()
            value_loss += weight * value.detach().item()
    return policy_loss, value_loss


def _compute_ddqn_loss_same_objectives(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        y_true_history,
        sigma_true_history,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_y_true_history,
        next_sigma_true_history,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
        *codein_states,
    ) = batch

    device = cfg.device
    batch_to_device_started_at = time.perf_counter()
    archive_mask = torch.as_tensor(build_row_mask(x_true), dtype=torch.bool, device=device)
    candidate_mask = torch.as_tensor(build_row_mask(x_sur), dtype=torch.bool, device=device)
    next_archive_mask = torch.as_tensor(build_row_mask(next_x_true), dtype=torch.bool, device=device)
    next_candidate_mask = torch.as_tensor(build_row_mask(next_x_sur), dtype=torch.bool, device=device)

    x_true_pad = pad_stack_rows(x_true)
    y_true_pad = pad_stack_rows(y_true)
    y_true_history_pad = pad_stack_rows(y_true_history)
    sigma_true_history_pad = pad_stack_rows(sigma_true_history)
    x_sur_pad = pad_stack_rows(x_sur)
    y_sur_pad = pad_stack_rows(y_sur)
    sigma_sur_pad = pad_stack_rows(sigma_sur)
    progress_arr = np.asarray(progress).reshape(-1, 1)
    lower_bound_pad = pad_stack_rows(lower_bound)
    upper_bound_pad = pad_stack_rows(upper_bound)

    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    next_x_true_pad = pad_stack_rows(next_x_true)
    next_y_true_pad = pad_stack_rows(next_y_true)
    next_y_true_history_pad = pad_stack_rows(next_y_true_history)
    next_sigma_true_history_pad = pad_stack_rows(next_sigma_true_history)
    next_x_sur_pad = pad_stack_rows(next_x_sur)
    next_y_sur_pad = pad_stack_rows(next_y_sur)
    next_sigma_sur_pad = pad_stack_rows(next_sigma_sur)
    next_progress_arr = np.asarray(next_progress).reshape(-1, 1)

    batch_cpu_nbytes = (
        int(x_true_pad.nbytes)
        + int(y_true_pad.nbytes)
        + int(y_true_history_pad.nbytes)
        + int(sigma_true_history_pad.nbytes)
        + int(x_sur_pad.nbytes)
        + int(y_sur_pad.nbytes)
        + int(sigma_sur_pad.nbytes)
        + int(progress_arr.nbytes)
        + int(lower_bound_pad.nbytes)
        + int(upper_bound_pad.nbytes)
        + int(next_x_true_pad.nbytes)
        + int(next_y_true_pad.nbytes)
        + int(next_y_true_history_pad.nbytes)
        + int(next_sigma_true_history_pad.nbytes)
        + int(next_x_sur_pad.nbytes)
        + int(next_y_sur_pad.nbytes)
        + int(next_sigma_sur_pad.nbytes)
        + int(next_progress_arr.nbytes)
        + int(archive_mask.numel())
        + int(candidate_mask.numel())
        + int(next_archive_mask.numel())
        + int(next_candidate_mask.numel())
    )

    x_true = to_tensor(x_true_pad, device)
    y_true = to_tensor(y_true_pad, device)
    y_true_history = to_tensor(y_true_history_pad, device)
    sigma_true_history = to_tensor(sigma_true_history_pad, device)
    x_sur = to_tensor(x_sur_pad, device)
    y_sur = to_tensor(y_sur_pad, device)
    sigma_sur = to_tensor(sigma_sur_pad, device)
    progress = to_tensor(progress_arr, device)
    lower_bound = to_tensor(lower_bound_pad, device)
    upper_bound = to_tensor(upper_bound_pad, device)
    next_x_true = to_tensor(next_x_true_pad, device)
    next_y_true = to_tensor(next_y_true_pad, device)
    next_y_true_history = to_tensor(next_y_true_history_pad, device)
    next_sigma_true_history = to_tensor(next_sigma_true_history_pad, device)
    next_x_sur = to_tensor(next_x_sur_pad, device)
    next_y_sur = to_tensor(next_y_sur_pad, device)
    next_sigma_sur = to_tensor(next_sigma_sur_pad, device)
    next_progress = to_tensor(next_progress_arr, device)
    batch_to_device_sec = time.perf_counter() - batch_to_device_started_at

    batch_gpu_nbytes = (
        _tensor_nbytes(x_true)
        + _tensor_nbytes(y_true)
        + _tensor_nbytes(y_true_history)
        + _tensor_nbytes(sigma_true_history)
        + _tensor_nbytes(x_sur)
        + _tensor_nbytes(y_sur)
        + _tensor_nbytes(sigma_sur)
        + _tensor_nbytes(progress)
        + _tensor_nbytes(lower_bound)
        + _tensor_nbytes(upper_bound)
        + _tensor_nbytes(actions)
        + _tensor_nbytes(rewards)
        + _tensor_nbytes(dones)
        + _tensor_nbytes(next_x_true)
        + _tensor_nbytes(next_y_true)
        + _tensor_nbytes(next_y_true_history)
        + _tensor_nbytes(next_sigma_true_history)
        + _tensor_nbytes(next_x_sur)
        + _tensor_nbytes(next_y_sur)
        + _tensor_nbytes(next_sigma_sur)
        + _tensor_nbytes(next_progress)
        + _tensor_nbytes(archive_mask)
        + _tensor_nbytes(candidate_mask)
        + _tensor_nbytes(next_archive_mask)
        + _tensor_nbytes(next_candidate_mask)
    )

    if codein_states:
        current_kwargs = _batch_codein_constraints(codein_states[0], device, include_offspring=True)
        next_kwargs = _batch_codein_constraints(codein_states[1], device, include_offspring=True)
    else:
        current_kwargs = {
            "decode_type": "q_greedy",
            **prediction_history_kwargs(agent, y_true_history, sigma_true_history),
        }
        next_kwargs = None

    out = agent(
        x_true=x_true,
        y_true=y_true,
        x_sur=x_sur,
        y_sur=y_sur,
        sigma_sur=sigma_sur,
        progress=progress,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        archive_mask=archive_mask,
        candidate_mask=candidate_mask,
        **current_kwargs,
    )

    q_values = out["q_values"]
    q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        if next_candidate_mask.size(1):
            next_inputs = dict(
                x_true=next_x_true,
                y_true=next_y_true,
                x_sur=next_x_sur,
                y_sur=next_y_sur,
                sigma_sur=next_sigma_sur,
                progress=next_progress,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                archive_mask=next_archive_mask,
                candidate_mask=next_candidate_mask,
            )
            target_kwargs = next_kwargs if codein_states else prediction_history_kwargs(
                target_agent, next_y_true_history, next_sigma_true_history
            )
            online_kwargs = next_kwargs if codein_states else prediction_history_kwargs(
                agent, next_y_true_history, next_sigma_true_history
            )
            if not codein_states:
                next_inputs["decode_type"] = "q_greedy"
            next_target = target_agent(**next_inputs, **target_kwargs)["q_values"]
            next_online = agent(**next_inputs, **online_kwargs)["q_values"]
            next_actions = next_online.masked_fill(~next_candidate_mask, -torch.inf).argmax(dim=1)
            next_q = next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + cfg.gamma * next_q.masked_fill(dones.bool(), 0.0)

    td_error = q_sa - target
    loss = nn.SmoothL1Loss()(q_sa, target.detach())
    metrics = {
        "q_mean": q_sa.detach().mean().item(),
        "q_std": q_sa.detach().std(unbiased=False).item() if q_sa.numel() > 1 else 0.0,
        "target_mean": target.detach().mean().item(),
        "td_error_mean": td_error.detach().mean().item(),
        "td_error_std": td_error.detach().std(unbiased=False).item() if td_error.numel() > 1 else 0.0,
        "reward_mean": rewards.mean().item(),
        "batch_to_device_sec": float(batch_to_device_sec),
        "batch_cpu_mb": float(batch_cpu_nbytes) / (1024.0 * 1024.0),
        "batch_gpu_mb": float(batch_gpu_nbytes) / (1024.0 * 1024.0),
        "archive_rows_mean": float(np.mean([np.asarray(arr).shape[0] for arr in batch[0]])),
        "candidate_rows_mean": float(np.mean([np.asarray(arr).shape[0] for arr in batch[4]])),
        "archive_rows_max": int(max(max(int(archive_mask.shape[1]), int(next_archive_mask.shape[1])), 0)),
        "candidate_rows_max": int(max(max(int(candidate_mask.shape[1]), int(next_candidate_mask.shape[1])), 0)),
    }
    return loss, metrics, len(x_true)


def compute_ddqn_loss(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        y_true_history,
        sigma_true_history,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_y_true_history,
        next_sigma_true_history,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
        *codein_states,
    ) = batch

    objective_counts = [int(np.asarray(arr).shape[1]) for arr in y_true]
    groups = {}
    for idx, n_obj in enumerate(objective_counts):
        groups.setdefault(n_obj, []).append(idx)

    total_count = 0
    total_q_mean = 0.0
    total_q_std = 0.0
    total_target_mean = 0.0
    total_td_error_mean = 0.0
    total_td_error_std = 0.0
    total_r_mean = 0.0
    total_batch_to_device_sec = 0.0
    total_batch_cpu_mb = 0.0
    total_batch_gpu_mb = 0.0
    total_archive_rows_mean = 0.0
    total_candidate_rows_mean = 0.0
    max_archive_rows = 0
    max_candidate_rows = 0
    weighted_loss = None
    group_sizes = []
    group_objectives = []

    batch_items = [
        x_true,
        y_true,
        y_true_history,
        sigma_true_history,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_y_true_history,
        next_sigma_true_history,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
        *codein_states,
    ]

    for n_obj, indices in groups.items():
        subbatch = []
        for item in batch_items:
            subbatch.append([item[i] for i in indices])

        group_loss, group_metrics, group_count = _compute_ddqn_loss_same_objectives(
            agent=agent,
            target_agent=target_agent,
            batch=tuple(subbatch),
            cfg=cfg,
        )

        total_count += int(group_count)
        group_sizes.append(int(group_count))
        group_objectives.append(int(n_obj))
        total_q_mean += float(group_metrics["q_mean"]) * float(group_count)
        total_q_std += float(group_metrics["q_std"]) * float(group_count)
        total_target_mean += float(group_metrics["target_mean"]) * float(group_count)
        total_td_error_mean += float(group_metrics["td_error_mean"]) * float(group_count)
        total_td_error_std += float(group_metrics["td_error_std"]) * float(group_count)
        total_r_mean += float(group_metrics["reward_mean"]) * float(group_count)
        total_batch_to_device_sec += float(group_metrics["batch_to_device_sec"])
        total_batch_cpu_mb += float(group_metrics["batch_cpu_mb"])
        total_batch_gpu_mb += float(group_metrics["batch_gpu_mb"])
        total_archive_rows_mean += float(group_metrics["archive_rows_mean"]) * float(group_count)
        total_candidate_rows_mean += float(group_metrics["candidate_rows_mean"]) * float(group_count)
        max_archive_rows = max(int(max_archive_rows), int(group_metrics["archive_rows_max"]))
        max_candidate_rows = max(int(max_candidate_rows), int(group_metrics["candidate_rows_max"]))

        scaled_loss = group_loss * (float(group_count) / float(len(objective_counts)))
        weighted_loss = scaled_loss if weighted_loss is None else (weighted_loss + scaled_loss)

    if weighted_loss is None or total_count <= 0:
        raise ValueError("Failed to build objective-shape groups for DDQN loss.")

    metrics = {
        "q_mean": total_q_mean / total_count,
        "q_std": total_q_std / total_count,
        "target_mean": total_target_mean / total_count,
        "td_error_mean": total_td_error_mean / total_count,
        "td_error_std": total_td_error_std / total_count,
        "reward_mean": total_r_mean / total_count,
        "batch_to_device_sec": total_batch_to_device_sec,
        "batch_cpu_mb": total_batch_cpu_mb,
        "batch_gpu_mb": total_batch_gpu_mb,
        "archive_rows_mean": total_archive_rows_mean / total_count,
        "candidate_rows_mean": total_candidate_rows_mean / total_count,
        "archive_rows_max": int(max_archive_rows),
        "candidate_rows_max": int(max_candidate_rows),
        "shape_group": len(group_sizes),
        "group_sizes": group_sizes,
        "group_objectives": group_objectives,
        "shape_group_detail": {
            int(n_obj): int(sz) for n_obj, sz in zip(group_objectives, group_sizes)
        },
    }
    return weighted_loss, metrics


def compute_env_reward(
    previous_archive_y,
    selected_y,
    ref_point,
    reward_scheme_id,
    reward_lambda=5.0,
    true_pareto_hv=None,
):
    previous_front = pareto_front(np.asarray(previous_archive_y, dtype=np.float32))
    selected_y = np.asarray(selected_y, dtype=np.float32)

    if int(reward_scheme_id) == 1:
        return float(
            reward_scheme_1(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
                reward_lambda=float(reward_lambda),
            )
        )
    if int(reward_scheme_id) == 2:
        if true_pareto_hv is None:
            raise ValueError("reward_scheme_2 requires true_pareto_hv.")
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
                reward_lambda=float(reward_lambda),
            )
        )
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme_id}")


def build_training_env_specs(problem_name, dim=None):
    heldout = str(problem_name).upper()
    if heldout not in {"MW", "DAS"}:
        raise ValueError("Training --problem must be MW or DAS.")
    benchmarks = DAS_PROBLEMS if heldout == "MW" else MW_PROBLEMS
    expected_dim = default_problem_dim(benchmarks[0])
    if dim is not None and int(dim) != expected_dim:
        raise ValueError(f"Training with --problem {heldout} uses {expected_dim} decision variables.")
    return [{"problem_name": name, "dim": expected_dim} for name in benchmarks]


def env_key(problem_name, dim):
    return f"{str(problem_name).upper()}-{int(dim)}D"


def effective_surrogate_label(cfg_like) -> str:
    return "gp"


class DiscSAEAEnv:
    def __init__(self, problem_name, dim, seed, cfg_dict, init_policy=None):
        self.problem_name = str(problem_name)
        self.dim = int(dim)
        self.seed = int(seed)
        self.cfg = dict(cfg_dict)
        self.problem = make_problem(self.problem_name, dim=self.dim)
        self.lower_bound = np.full(self.dim, float(self.problem.lower), dtype=np.float32)
        self.upper_bound = np.full(self.dim, float(self.problem.upper), dtype=np.float32)
        self.max_steps = max(1, int(self.cfg["max_fe"]) - int(self.cfg["init_size"]))
        self.eval_batch_size = max(1, int(self.cfg.get("eval_batch_size", TrainConfig.eval_batch_size)))
        self._pool_eval_count = 0
        self.t = 0
        self.archive_x = None
        self.archive_y = None
        self.archive_y_history = None
        self.archive_sigma_history = None
        self.offspring_x = None
        self.offspring_y = None
        self.offspring_sigma = None
        self.offspring_info = {}
        self.ref_point = None
        self.nsga2_problem = None
        self.init_hv = None
        self.true_pareto_hv = None
        self.surrogate = None
        self._surrogate_dirty = False
        self.init_policy = init_policy

    def _progress(self):
        max_fe = int(self.cfg["max_fe"])
        if self.init_policy is not None:
            return (int(self.cfg["init_size"]) + self.t) / max_fe
        current_fe = min(int(self.cfg["init_size"]) + int(self.t), max_fe - 1)
        return float(current_fe) / float(max_fe)

    def _surrogate_cfg(self):
        cfg_local = dict(self.cfg)
        cfg_local["seed"] = int(self.seed) + int(self.t)
        cfg_local["problem_name"] = str(self.problem_name)
        return cfg_local

    def _fit_surrogate(self):
        self.surrogate = build_surrogate_from_cfg(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
        )
        self._surrogate_dirty = False
        return self.surrogate

    def _ensure_surrogate_ready(self):
        if self.surrogate is None or bool(self._surrogate_dirty):
            self._fit_surrogate()
        return self.surrogate

    def _refresh_offspring(self):
        seed_x = None
        if self.init_policy is not None:
            self.seed_count = min(
                int(self.cfg.get("num_seeds", TrainConfig.num_seeds)),
                len(self.archive_x), int(self.cfg.get("offspring_size", TrainConfig.offspring_size)),
            )
            state = self._archive_policy_state()
            with torch.no_grad():
                indices, grades, log_probs, _ = self.init_policy.choose_seeds(
                    sample=True, **tensor_policy_state(state, self.cfg.get("rollout_device", "cpu")),
                )
            self.init_decisions.append((self.t, state, grades[0].cpu().numpy(), log_probs[0].cpu().numpy()))
            seed_x = self.archive_x[indices[0].cpu().numpy()]
        surrogate, offspring_x, offspring_y, offspring_sigma, offspring_info = generate_offspring_pool(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
            nsga2_problem=self.nsga2_problem,
            seed=int(self.seed) + int(self.t),
            seed_x=seed_x,
        )
        self.surrogate = surrogate
        self.offspring_x = np.asarray(offspring_x, dtype=np.float32)
        self.offspring_y = np.asarray(offspring_y, dtype=np.float32)
        self.offspring_sigma = np.asarray(offspring_sigma, dtype=np.float32)
        self.offspring_info = dict(offspring_info)
        if self.init_policy is not None:
            self.g_sur_mean, self.g_sur_std = self._predict_constraints()
        self._pool_eval_count = 0

    def _predict_constraints(self):
        if self.archive_g.shape[1] == 0:
            empty = np.empty((len(self.offspring_x), 0), dtype=np.float32)
            return empty, empty.copy()
        surrogate = build_surrogate_from_cfg(self._surrogate_cfg(), self.archive_x, self.archive_g)
        return surrogate.predict_mean(self.offspring_x), surrogate.predict_std(self.offspring_x)

    def _remove_offspring_candidate(self, selected_idx: int) -> None:
        n_candidates = int(np.asarray(self.offspring_x).shape[0])
        if self.init_policy is not None:
            self.g_sur_mean = np.delete(self.g_sur_mean, selected_idx, axis=0)
            self.g_sur_std = np.delete(self.g_sur_std, selected_idx, axis=0)
        if n_candidates <= 1:
            self.offspring_x = np.empty((0, self.dim), dtype=np.float32)
            n_obj = int(np.asarray(self.archive_y).shape[1])
            self.offspring_y = np.empty((0, n_obj), dtype=np.float32)
            self.offspring_sigma = np.empty((0, n_obj), dtype=np.float32)
            return
        keep_mask = np.ones(n_candidates, dtype=bool)
        keep_mask[int(selected_idx)] = False
        self.offspring_x = np.asarray(self.offspring_x, dtype=np.float32)[keep_mask]
        self.offspring_y = np.asarray(self.offspring_y, dtype=np.float32)[keep_mask]
        self.offspring_sigma = np.asarray(self.offspring_sigma, dtype=np.float32)[keep_mask]

    def _archive_policy_state(self):
        return {
            "x_true": self.archive_x.copy(),
            "y_true": self.archive_y.copy(),
            "g_true": self.archive_g.copy(),
            "constraint_scale": self.constraint_scale.copy(),
            "seed_count": self.seed_count,
            "progress": np.array([self._progress()], dtype=np.float32),
            "lower_bound": self.lower_bound.copy(),
            "upper_bound": self.upper_bound.copy(),
        }

    def _build_state(self):
        prediction_history = (
            (self.archive_y_history.copy(), self.archive_sigma_history.copy())
            if self.init_policy is None else (np.empty(0, dtype=np.float32),) * 2
        )
        state = {
            "x_true": self.archive_x.copy(),
            "y_true": self.archive_y.copy(),
            "g_true": self.archive_g.copy(),
            "constraint_scale": self.constraint_scale.copy(),
            "y_true_history": prediction_history[0],
            "sigma_true_history": prediction_history[1],
            "x_sur": np.asarray(self.offspring_x, dtype=np.float32).copy(),
            "y_sur": np.asarray(self.offspring_y, dtype=np.float32).copy(),
            "sigma_sur": np.asarray(self.offspring_sigma, dtype=np.float32).copy(),
            "progress": np.array([self._progress()], dtype=np.float32),
            "lower_bound": self.lower_bound.copy(),
            "upper_bound": self.upper_bound.copy(),
            "offspring_info": dict(self.offspring_info),
        }
        if self.init_policy is not None:
            state.update(
                seed_count=self.seed_count, g_sur_mean=self.g_sur_mean.copy(), g_sur_std=self.g_sur_std.copy(),
            )
        return state

    def reset(self):
        self.t = 0
        self.archive_x = latin_hypercube_sample(
            n_samples=int(self.cfg["init_size"]),
            dim=self.dim,
            lower=self.problem.lower,
            upper=self.problem.upper,
            seed=self.seed,
        )
        self.archive_y, self.archive_g = evaluate_objectives_and_constraints(self.problem, self.archive_x)
        self.constraint_scale = initial_constraint_scale(self.archive_g)
        if self.init_policy is None:
            history_steps = int(self.cfg.get("prediction_history_steps", 40))
            self.archive_y_history = np.repeat(self.archive_y[:, None, :], history_steps, axis=1)
            self.archive_sigma_history = np.zeros_like(self.archive_y_history)
        else:
            self.init_decisions = []
        self.ref_point = get_reference_point(self.archive_y)
        self.init_hv = float(hypervolume(self.archive_y, self.ref_point))
        self.true_pareto_hv = (
            hypervolume(reference_front(self.problem_name), self.ref_point)
            if int(self.cfg.get("reward_scheme", 1)) == 2 else None
        )
        self.nsga2_problem = make_nsga2_problem_adapter(self.problem, int(self.archive_y.shape[1]))
        # Pretrain the surrogate on the initial archive, then generate the first pool.
        self._fit_surrogate()
        self._refresh_offspring()
        return self._build_state()

    def step(self, action_idx, state):
        del state
        chosen_idx = int(np.clip(int(action_idx), 0, int(self.offspring_x.shape[0]) - 1))
        previous_archive_y = np.asarray(self.archive_y, dtype=np.float32).copy()
        chosen_x = self.offspring_x[chosen_idx : chosen_idx + 1]
        chosen_pred = np.asarray(self.offspring_y[chosen_idx : chosen_idx + 1], dtype=np.float32)
        chosen_sigma = np.asarray(self.offspring_sigma[chosen_idx : chosen_idx + 1], dtype=np.float32)
        chosen_y, chosen_g = evaluate_objectives_and_constraints(self.problem, chosen_x)
        self.archive_g = np.vstack((self.archive_g, chosen_g))
        if self.init_policy is None:
            history_steps = self.archive_y_history.shape[1]
            new_y_history = np.repeat(chosen_pred[:, None, :], history_steps, axis=1).astype(np.float32)
            new_sigma_history = np.repeat(chosen_sigma[:, None, :], history_steps, axis=1).astype(np.float32)
            eval_step = min(self.t, history_steps - 1)
            new_y_history[:, eval_step:, :] = chosen_y[:, None, :]
            new_sigma_history[:, eval_step:, :] = 0.0
            self.archive_y_history = np.vstack((self.archive_y_history, new_y_history))
            self.archive_sigma_history = np.vstack((self.archive_sigma_history, new_sigma_history))

        self.archive_x = np.vstack([self.archive_x, chosen_x]).astype(np.float32)
        self.archive_y = np.vstack([self.archive_y, chosen_y]).astype(np.float32)

        reward = compute_env_reward(
            previous_archive_y=previous_archive_y,
            selected_y=chosen_y,
            ref_point=self.ref_point,
            reward_scheme_id=int(self.cfg["reward_scheme"]),
            reward_lambda=float(self.cfg.get("reward_lambda", 5.0)),
            true_pareto_hv=self.true_pareto_hv,
        )

        self.t += 1
        done = self.t >= self.max_steps
        self._surrogate_dirty = bool(not done)
        if done and self.init_policy is not None:
            self.offspring_x = self.offspring_x[:0]
            self.offspring_y = self.offspring_y[:0]
            self.offspring_sigma = self.offspring_sigma[:0]
            self.g_sur_mean = self.g_sur_mean[:0]
            self.g_sur_std = self.g_sur_std[:0]
        if not done:
            self._pool_eval_count += 1
            should_refresh = (
                int(self._pool_eval_count) >= int(self.eval_batch_size)
                or int(np.asarray(self.offspring_x).shape[0]) <= 1
            )
            if should_refresh:
                self._refresh_offspring()
            else:
                self._remove_offspring_candidate(chosen_idx)
        return self._build_state(), float(reward), bool(done)

    def current_hv(self):
        return float(hypervolume(np.asarray(self.archive_y, dtype=np.float32), self.ref_point))


def _transition_from_states(state, action, reward, next_state, done):
    transition = (
        state["x_true"],
        state["y_true"],
        state["y_true_history"],
        state["sigma_true_history"],
        state["x_sur"],
        state["y_sur"],
        state["sigma_sur"],
        float(state["progress"][0]),
        state["lower_bound"],
        state["upper_bound"],
        int(action),
        float(reward),
        next_state["x_true"],
        next_state["y_true"],
        next_state["y_true_history"],
        next_state["sigma_true_history"],
        next_state["x_sur"],
        next_state["y_sur"],
        next_state["sigma_sur"],
        float(next_state["progress"][0]),
        float(done),
    )
    if "g_sur_mean" in state:
        keys = ("g_true", "g_sur_mean", "g_sur_std", "constraint_scale", "seed_count")
        transition += ({key: state[key] for key in keys}, {key: next_state[key] for key in keys})
    return transition


def _rollout_storage_metrics(transitions):
    transition_sizes = [int(_estimate_transition_nbytes(item)) for item in transitions]
    archive_rows = [int(np.asarray(item[0]).shape[0]) for item in transitions]
    candidate_rows = [int(np.asarray(item[4]).shape[0]) for item in transitions]
    return {
        "archive_rows_mean": float(np.mean(archive_rows)) if archive_rows else 0.0,
        "archive_rows_max": int(max(archive_rows)) if archive_rows else 0,
        "candidate_rows_mean": float(np.mean(candidate_rows)) if candidate_rows else 0.0,
        "candidate_rows_max": int(max(candidate_rows)) if candidate_rows else 0,
        "transition_bytes_total": int(sum(transition_sizes)),
        "transition_bytes_mean": float(np.mean(transition_sizes)) if transition_sizes else 0.0,
        "transition_bytes_max": int(max(transition_sizes)) if transition_sizes else 0,
    }


def _rollout_episode_impl(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    task_started_at = time.perf_counter()
    worker_pid = os.getpid()
    device = str(cfg_dict.get("rollout_device", "cpu"))
    seed_everything(int(seed), include_cuda=str(device).startswith("cuda"))
    agent_name = str(cfg_dict.get("agent_name", "codein")).strip().lower()
    agent_cls = resolve_agent_cls(agent_name)
    agent_init_kwargs = build_agent_init_kwargs(cfg_dict, epsilon=epsilon)
    agent = agent_cls(**agent_init_kwargs)
    agent = agent.to(device)

    agent.load_state_dict(state_dict_cpu)
    agent.eval()
    codein = isinstance(agent, CodeIn)

    env = DiscSAEAEnv(
        problem_name=problem_name,
        dim=int(dim),
        seed=int(seed),
        cfg_dict=cfg_dict,
        init_policy=agent if codein else None,
    )
    state = env.reset()
    total_reward = 0.0
    init_hv = float(env.init_hv)
    transitions = []
    disc_steps = 0

    done = False
    while not done:
        with torch.no_grad():
            if codein:
                out = agent(**tensor_policy_state(state, device, include_offspring=True))
                action = int(agent.choose_candidate(
                    out, sample=True, temperature=float(cfg_dict.get("temperature", TrainConfig.temperature))
                ).item())
            else:
                out = agent(
                    x_true=to_tensor(state["x_true"][None, ...], device),
                    y_true=to_tensor(state["y_true"][None, ...], device),
                    x_sur=to_tensor(state["x_sur"][None, ...], device),
                    y_sur=to_tensor(state["y_sur"][None, ...], device),
                    sigma_sur=to_tensor(state["sigma_sur"][None, ...], device),
                    progress=to_tensor(state["progress"].reshape(1, 1), device),
                    lower_bound=to_tensor(state["lower_bound"][None, ...], device),
                    upper_bound=to_tensor(state["upper_bound"][None, ...], device),
                    decode_type=str(cfg_dict.get("policy_mode", "epsilon_greedy")),
                    epsilon=epsilon,
                    **prediction_history_kwargs(
                        agent,
                        to_tensor(state["y_true_history"][None, ...], device),
                        to_tensor(state["sigma_true_history"][None, ...], device),
                    ),
                )
                action = select_action_from_output(out)
        disc_steps += 1
        next_state, reward, done = env.step(action, state)

        transition = _transition_from_states(state, action, reward, next_state, done)
        transitions.append(transition)

        total_reward += reward
        state = next_state

    init_rollout = []
    if codein:
        returns = [0.0] * len(transitions)
        discounted = 0.0
        for i in range(len(transitions) - 1, -1, -1):
            discounted = transitions[i][11] + float(cfg_dict.get("gamma", TrainConfig.gamma)) * discounted
            returns[i] = discounted
        init_rollout = [
            (init_state, grades, log_probs, returns[step])
            for step, init_state, grades, log_probs in env.init_decisions
        ]
    elif bool(cfg_dict.get("reward_norm", False)):
        transitions = _normalize_episode_transition_rewards(transitions)

    task_finished_at = time.perf_counter()
    return {
        "transitions": transitions,
        "init_rollout": init_rollout,
        "episode_reward": float(total_reward),
        "episode_steps": int(len(transitions)),
        "logging_reward": float(total_reward),
        "logging_steps": int(len(transitions)),
        "env_key": env_key(problem_name, dim),
        "init_hv": init_hv,
        "final_hv": float(env.current_hv()),
        **_rollout_storage_metrics(transitions),
        "worker_pid": int(worker_pid),
        "disc_steps": int(disc_steps),
        "task_started_at": float(task_started_at),
        "task_finished_at": float(task_finished_at),
        "task_duration_sec": float(task_finished_at - task_started_at),
    }


def rollout_episode_task_local(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    return _rollout_episode_impl(
        state_dict_cpu=state_dict_cpu,
        cfg_dict=cfg_dict,
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epsilon=epsilon,
    )


def summarize_parallel_rollout(results):
    if not results:
        return None
    started = np.asarray([float(r["task_started_at"]) for r in results], dtype=np.float64)
    finished = np.asarray([float(r["task_finished_at"]) for r in results], dtype=np.float64)
    durations = np.asarray([float(r["task_duration_sec"]) for r in results], dtype=np.float64)
    pids = [int(r["worker_pid"]) for r in results]
    disc_steps = int(sum(int(r.get("disc_steps", 0)) for r in results))
    overlap_wall_sec = float(max(finished.max() - started.min(), 0.0))
    sum_task_sec = float(durations.sum())
    parallelism = float(sum_task_sec / max(overlap_wall_sec, 1e-12))
    return {
        "tasks": int(len(results)),
        "unique_pids": int(len(set(pids))),
        "pid_list": sorted(set(pids)),
        "disc_steps": disc_steps,
        "overlap_wall_sec": overlap_wall_sec,
        "sum_task_sec": sum_task_sec,
        "mean_task_sec": float(durations.mean()),
        "min_task_sec": float(durations.min()),
        "max_task_sec": float(durations.max()),
        "parallelism_est": parallelism,
    }


def _prepare_training_run_layout(log_dir: str, log_filename: str) -> tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)

    run_stem = os.path.splitext(log_filename)[0]
    run_dir = os.path.join(log_dir, run_stem)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, os.path.join(run_dir, log_filename)


def save_training_checkpoint(
    agent,
    cfg,
    problem_name,
    epoch,
    mean_reward,
    best_reward,
    best_state_dict=None,
):
    os.makedirs(cfg.weight_dir, exist_ok=True)
    rs_tag = f"rs{int(cfg.reward_scheme)}"
    hidden_tag = f"h{int(cfg.hidden_dim)}"
    problem_tag = str(problem_name).lower()
    agent_tag = str(getattr(cfg, "agent_log_key", cfg.agent_name)).lower()
    file_prefix = f"{agent_tag}_problem_{problem_tag}_{rs_tag}_{hidden_tag}"

    if mean_reward > best_reward:
        best_path = os.path.join(cfg.weight_dir, f"{file_prefix}_best_reward.pth")
        state_dict_to_save = best_state_dict if best_state_dict is not None else agent.state_dict()
        torch.save(
            {
                "epoch": int(epoch),
                "problem_name": str(problem_name),
                "reward_scheme": int(cfg.reward_scheme),
                "mean_reward": float(mean_reward),
                "state_dict": state_dict_to_save,
            },
            best_path,
        )
        best_reward = float(mean_reward)

    epoch_path = os.path.join(cfg.weight_dir, f"{file_prefix}_epoch_{int(epoch)}.pth")
    torch.save(
        {
            "epoch": int(epoch),
            "problem_name": str(problem_name),
            "reward_scheme": int(cfg.reward_scheme),
            "mean_reward": float(mean_reward),
            "state_dict": agent.state_dict(),
        },
        epoch_path,
    )

    return best_reward


def _normalize_episode_transition_rewards(transitions):
    if len(transitions) == 0:
        return transitions

    reward_idx = 11
    rewards = np.asarray([float(transition[reward_idx]) for transition in transitions], dtype=np.float32)
    reward_mean = float(np.mean(rewards))
    reward_std = float(np.std(rewards))
    denom = reward_std if reward_std > 1e-8 else 1.0
    normalized_rewards = (rewards - reward_mean) / denom

    normalized_transitions = []
    for idx, transition in enumerate(transitions):
        transition_list = list(transition)
        transition_list[reward_idx] = float(normalized_rewards[idx])
        normalized_transitions.append(tuple(transition_list))
    return normalized_transitions


def train_codein(
    problem_name="MW",
    dim=None,
    seed=0,
    epoch=None,
    gamma=None,
    reward_scheme=1,
    reward_norm=False,
    surrogate_model="gp",
    solver="moead_ego",
    num_workers=None,
    eval_batch_size=TrainConfig.eval_batch_size,
    updates_per_epoch=None,
    hidden_dim=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cpu",
    use_ray=False,
    agent_name="codein",
    cuda_cleanup_before_update=None,
    cuda_cleanup_after_update=None,
    num_seeds=TrainConfig.num_seeds,
    temperature=TrainConfig.temperature,
    value_coef=TrainConfig.value_coef,
    q_coef=TrainConfig.q_coef,
    init_fe=None,
    max_fe=None,
):
    cfg = TrainConfig()
    if init_fe is not None:
        cfg.init_size = int(init_fe)
    if max_fe is not None:
        cfg.max_fe = int(max_fe)
    if cfg.init_size < 2 or cfg.max_fe <= cfg.init_size:
        raise ValueError("Evaluation budget requires 2 <= init_fe < max_fe.")
    cfg.seed = int(seed)
    if epoch is not None:
        cfg.train_iters = int(epoch)
    if gamma is not None:
        cfg.gamma = float(gamma)
    cfg.reward_scheme = int(reward_scheme)
    cfg.reward_lambda = float(resolve_default_reward_lambda(cfg.reward_scheme))
    cfg.reward_norm = bool(reward_norm)
    cfg.surrogate_model = str(surrogate_model).lower()
    cfg.solver = str(solver).lower()
    default_cleanup = True
    cfg.cuda_cleanup_before_update = (
        bool(default_cleanup)
        if cuda_cleanup_before_update is None
        else bool(cuda_cleanup_before_update)
    )
    cfg.cuda_cleanup_after_update = (
        bool(default_cleanup)
        if cuda_cleanup_after_update is None
        else bool(cuda_cleanup_after_update)
    )
    cfg.heldout_problem = str(problem_name).upper()
    cfg.eval_batch_size = int(eval_batch_size)
    if updates_per_epoch is not None:
        cfg.updates_per_epoch = int(updates_per_epoch)
    if hidden_dim is not None:
        cfg.hidden_dim = int(hidden_dim)
    if device is not None:
        cfg.device = str(device)
    cfg.rollout_device = str(rollout_device)
    cfg.surrogate_device = str(surrogate_device)
    cfg.agent_name = str(agent_name).lower()
    cfg.num_seeds = int(num_seeds)
    cfg.temperature = float(temperature)
    cfg.value_coef, cfg.q_coef = float(value_coef), float(q_coef)
    codein = cfg.agent_name == "codein"
    if codein:
        cfg.policy_mode = "softmax_sample"
        cfg.reward_norm = False
        if cfg.num_seeds < 1 or not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError("num_seeds must be positive and temperature must be finite and positive.")
        if any(not math.isfinite(coef) or coef < 0 for coef in (cfg.value_coef, cfg.q_coef)):
            raise ValueError("value_coef and q_coef must be finite and nonnegative.")
        if cfg.updates_per_epoch < 1:
            raise ValueError("CodeIn requires at least one update per rollout epoch.")
    if num_workers is not None:
        cfg.num_workers = int(num_workers)
    if cfg.surrogate_model != "gp":
        raise ValueError(f"Unsupported surrogate_model: {cfg.surrogate_model}. Expected 'gp'.")
    if cfg.solver != "moead_ego":
        raise ValueError(f"Unsupported trainer solver: {cfg.solver}. Expected 'moead_ego'.")
    env_specs = build_training_env_specs(cfg.heldout_problem, dim)
    if int(cfg.num_workers) <= 0:
        raise ValueError(f"num_workers must be positive, got {cfg.num_workers}.")
    if int(cfg.train_iters) <= 0:
        raise ValueError(f"epoch must be positive, got {cfg.train_iters}.")
    if int(cfg.eval_batch_size) <= 0:
        raise ValueError(f"batch must be positive, got {cfg.eval_batch_size}.")
    actual_num_workers = min(int(cfg.num_workers), len(env_specs))
    seed_everything(cfg.seed, include_cuda=True)
    cfg_dict = cfg.__dict__.copy()
    os.makedirs("training_logs", exist_ok=True)
    cfg.agent_log_key = str(cfg.agent_name)
    log_dir = os.path.join("training_logs", cfg.agent_log_key)
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = f"{cfg.agent_log_key}_train"
    agent_cls = resolve_agent_cls(cfg.agent_name)
    agent = agent_cls(**build_agent_init_kwargs(cfg, epsilon=cfg.epsilon_start)).to(cfg.device)

    target_agent = copy.deepcopy(agent).to(cfg.device)
    target_agent.eval()

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)
    best_reward = -float("inf")
    log_path = os.path.join(
        log_dir,
        f"{log_prefix}_{cfg.heldout_problem.lower()}_set1_{ts}.txt",
    )
    run_dir, log_path = _prepare_training_run_layout(log_dir, os.path.basename(log_path))
    cfg.weight_dir = run_dir
    os.makedirs(cfg.weight_dir, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)

    def _write_log_file_line(msg: str) -> None:
        log_fp.write(str(msg) + "\n")

    def log(msg):
        print(msg)
        _write_log_file_line(str(msg))

    executor = None
    ray_rollout_remote = None
    ray_mod = None
    if bool(use_ray):
        try:
            import ray as ray_mod  # type: ignore
        except Exception as exc:
            raise ImportError("Ray backend is unavailable.") from exc
        if not ray_mod.is_initialized():
            ray_mod.init(num_cpus=actual_num_workers, ignore_reinit_error=True)
        ray_rollout_remote = ray_mod.remote(num_cpus=1)(rollout_episode_task_local)

    logged_epoch_target = int(cfg.train_iters)

    config_parts = [
        f"seed={cfg.seed}",
        f"heldout={cfg.heldout_problem}",
        "training_set=1",
        f"envs={len(env_specs)}",
        f"workers={actual_num_workers}",
        f"init_fe={cfg.init_size}",
        f"max_fe={cfg.max_fe}",
        f"reward_scheme={cfg.reward_scheme}",
        f"reward_lambda={cfg.reward_lambda:.4f}",
        f"reward_norm={int(bool(cfg.reward_norm))}",
        f"agent={cfg.agent_name}",
        f"policy={cfg.policy_mode}",
        f"solver={cfg.solver}",
        f"surrogate={effective_surrogate_label(cfg)}",
        f"hidden_dim={cfg.hidden_dim}",
        f"sampling_backend={'ray' if use_ray else 'process_pool'}",
        f"epochs={logged_epoch_target}",
        f"epochs_to_run={cfg.train_iters}",
        f"saea_steps={cfg.saea_steps}",
        f"eval_batch_size={cfg.eval_batch_size}",
        f"updates_per_epoch={cfg.updates_per_epoch}",
        f"train_device={cfg.device}",
        f"rollout_device={cfg.rollout_device}",
        f"surrogate_device={cfg.surrogate_device}",
        f"lr={cfg.lr:.1e}",
        f"batch_size={cfg.batch_size}",
        f"replay_size={cfg.replay_size}",
        f"gamma={cfg.gamma:.4f}",
        "target_update=soft",
        f"target_soft_tau={cfg.target_soft_tau:.4f}",
        f"log_path={log_path}",
    ]
    if bool(cfg.cuda_cleanup_before_update):
        config_parts.append(f"cuda_cleanup_before_update={int(bool(cfg.cuda_cleanup_before_update))}")
    if bool(cfg.cuda_cleanup_after_update):
        config_parts.append(f"cuda_cleanup_after_update={int(bool(cfg.cuda_cleanup_after_update))}")
    if codein:
        config_parts.extend([
            f"num_seeds={cfg.num_seeds}", f"temperature={cfg.temperature:g}",
            f"value_coef={cfg.value_coef:g}", f"q_coef={cfg.q_coef:g}",
        ])
    training_config_message = (
        f"training | heldout = {cfg.heldout_problem} | set = 1 | "
        f"agent = {cfg.agent_name} | solver = {cfg.solver} | surrogate = {effective_surrogate_label(cfg)} | "
        f"epochs = {logged_epoch_target} | workers = {actual_num_workers}"
    )
    log(training_config_message)
    replay = ReplayBuffer(cfg.replay_size)
    for it in range(cfg.train_iters):
        epoch = int(it) + 1
        epoch_solver = cfg.solver
        epsilon = epsilon_by_iter(it, cfg)
        if not codein:
            replay = ReplayBuffer(cfg.replay_size)
        epoch_cfg_dict = cfg_dict.copy()
        epoch_cfg_dict["solver"] = str(epoch_solver)

        exploration = f"temperature = {cfg.temperature:g}" if codein else f"epsilon = {epsilon:.3f}"
        log(
            f"epoch {epoch}/{logged_epoch_target} | start | "
            f"solver = {epoch_solver} | {exploration}"
        )
        state_cpu = clone_state_dict_cpu(agent)
        pre_update_state_dict = copy.deepcopy(agent.state_dict())
        if not bool(use_ray) and executor is None:
            executor = ProcessPoolExecutor(max_workers=actual_num_workers)
        futures = []
        for env_idx, spec in enumerate(env_specs):
            for ep in range(int(cfg.episodes_per_worker)):
                seed = int(cfg.seed) + 100000 * epoch + 1000 * env_idx + ep
                if bool(use_ray):
                    futures.append(
                        ray_rollout_remote.remote(
                            state_dict_cpu=state_cpu,
                            cfg_dict=epoch_cfg_dict,
                            problem_name=spec["problem_name"],
                            dim=int(spec["dim"]),
                            seed=int(seed),
                            epsilon=epsilon,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            rollout_episode_task_local,
                            state_cpu,
                            epoch_cfg_dict,
                            spec["problem_name"],
                            int(spec["dim"]),
                            int(seed),
                            epsilon,
                        )
                    )
        if len(futures) == 0:
            raise ValueError("No rollout tasks were created.")
        if bool(use_ray):
            results = ray_mod.get(futures)
        else:
            results = []
            for future in futures:
                results.append(future.result())
            executor.shutdown(wait=True)
            executor = None
        parallel_stats = summarize_parallel_rollout(results)
        if parallel_stats is not None:
            log(
                f"[Epoch {epoch:04d}] interact parallel | "
                f"tasks={parallel_stats['tasks']} | "
                f"unique_pids={parallel_stats['unique_pids']} | "
                f"pid_list={parallel_stats['pid_list']} | "
                f"overlap_wall_sec={parallel_stats['overlap_wall_sec']:.3f} | "
                f"sum_task_sec={parallel_stats['sum_task_sec']:.3f} | "
                f"mean_task_sec={parallel_stats['mean_task_sec']:.3f} | "
                f"min_task_sec={parallel_stats['min_task_sec']:.3f} | "
                f"max_task_sec={parallel_stats['max_task_sec']:.3f} | "
                f"parallelism_est={parallel_stats['parallelism_est']:.2f}"
            )

        per_env_stats = {}
        replay_payload_bytes = 0
        replay_transition_bytes_mean = []
        replay_transition_bytes_max = []
        replay_archive_rows_mean = []
        replay_archive_rows_max = []
        replay_candidate_rows_mean = []
        replay_candidate_rows_max = []
        for result in results:
            replay.extend(result["transitions"])
            replay_payload_bytes += int(result.get("transition_bytes_total", 0))
            replay_transition_bytes_mean.append(float(result.get("transition_bytes_mean", 0.0)))
            replay_transition_bytes_max.append(int(result.get("transition_bytes_max", 0)))
            replay_archive_rows_mean.append(float(result.get("archive_rows_mean", 0.0)))
            replay_archive_rows_max.append(int(result.get("archive_rows_max", 0)))
            replay_candidate_rows_mean.append(float(result.get("candidate_rows_mean", 0.0)))
            replay_candidate_rows_max.append(int(result.get("candidate_rows_max", 0)))
            bucket = per_env_stats.setdefault(
                result["env_key"],
                {
                    "rewards": [],
                    "reward_per_fe": [],
                    "init_hv": [],
                    "final_hv": [],
                },
            )
            ep_reward = float(result.get("logging_reward", result["episode_reward"]))
            ep_steps = max(int(result.get("logging_steps", result.get("episode_steps", 0))), 1)
            bucket["rewards"].append(ep_reward)
            bucket["reward_per_fe"].append(ep_reward / float(ep_steps))
            bucket["init_hv"].append(float(result["init_hv"]))
            bucket["final_hv"].append(float(result["final_hv"]))
        per_env_summaries = {}
        for key, stats in sorted(per_env_stats.items()):
            if len(stats["rewards"]) == 0:
                continue
            per_env_summaries[key] = {
                "mean_reward": float(np.mean(stats["rewards"])),
                "mean_reward_per_fe": float(np.mean(stats["reward_per_fe"])),
                "init_hv": float(np.mean(stats["init_hv"])),
                "final_hv": float(np.mean(stats["final_hv"])),
            }
        reward_metric_name = "mean reward/FE"
        mean_ep_reward = float(np.mean([
            value["mean_reward_per_fe"] for value in per_env_summaries.values()
        ])) if per_env_summaries else 0.0
        if bool(getattr(cfg, "cuda_cleanup_before_update", False)):
            _cleanup_cuda_cache(cfg.device, rounds=3, sleep_sec=1.0)
        update_start_time = time.perf_counter()

        if len(replay) < cfg.batch_size and not codein:
            log(
                f"epoch {epoch}/{logged_epoch_target} | reward/FE = {mean_ep_reward:.4f} | "
                f"replay = {len(replay)} | update = skipped"
            )
            if mean_ep_reward > best_reward:
                log(f"new best {reward_metric_name} at epoch {epoch}: {mean_ep_reward:.4f}")
            best_reward = save_training_checkpoint(
                agent,
                cfg,
                cfg.heldout_problem,
                epoch,
                mean_ep_reward,
                best_reward,
                best_state_dict=pre_update_state_dict,
            )
            continue

        update_metrics_list = []
        total_minibatch_sample_cpu_sec = 0.0
        total_minibatch_to_gpu_sec = 0.0
        # Recompute the sampled grades under the rollout policy before changing any weights.
        if codein:
            agent.eval()
            decisions = [decision for result in results for decision in result["init_rollout"]]
        else:
            agent.train()
        for update_idx in range(int(cfg.updates_per_epoch)):
            minibatch_sample_started_at = time.perf_counter()
            batch = replay.sample(min(cfg.batch_size, len(replay)))
            minibatch_sample_cpu_sec = time.perf_counter() - minibatch_sample_started_at
            loss, ddqn_metrics = compute_ddqn_loss(agent, target_agent, batch, cfg)
            total_minibatch_sample_cpu_sec += float(minibatch_sample_cpu_sec)
            total_minibatch_to_gpu_sec += float(ddqn_metrics.get("batch_to_device_sec", 0.0))

            optimizer.zero_grad()
            if codein:
                (cfg.q_coef * loss).backward()
                if update_idx == 0:
                    init_loss, value_loss = backward_codein_init(agent, decisions, cfg)
                    total_loss = init_loss + cfg.value_coef * value_loss + cfg.q_coef * loss.detach().item()
                    log(f"[Epoch {epoch:04d}] init | decisions={len(decisions)} | "
                        f"policy_loss={init_loss:.6f} | value_loss={value_loss:.6f} | total_loss={total_loss:.6f}")
            else:
                loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0))
            optimizer.step()
            soft_update_target_network(target_agent, agent, cfg.target_soft_tau)

            update_metrics = {
                "td_loss": float(loss.item()),
                "grad_norm": float(grad_norm),
                "q_mean": float(ddqn_metrics["q_mean"]),
                "q_std": float(ddqn_metrics["q_std"]),
                "target_mean": float(ddqn_metrics["target_mean"]),
                "td_error_mean": float(ddqn_metrics["td_error_mean"]),
                "td_error_std": float(ddqn_metrics["td_error_std"]),
                "reward_mean": float(ddqn_metrics["reward_mean"]),
                "batch_cpu_mb": float(ddqn_metrics.get("batch_cpu_mb", 0.0)),
                "batch_gpu_mb": float(ddqn_metrics.get("batch_gpu_mb", 0.0)),
                "archive_rows_mean": float(ddqn_metrics.get("archive_rows_mean", 0.0)),
                "candidate_rows_mean": float(ddqn_metrics.get("candidate_rows_mean", 0.0)),
                "archive_rows_max": int(ddqn_metrics.get("archive_rows_max", 0)),
                "candidate_rows_max": int(ddqn_metrics.get("candidate_rows_max", 0)),
                "shape_group": int(ddqn_metrics["shape_group"]),
                "group_sizes": list(ddqn_metrics["group_sizes"]),
                "shape_group_detail": dict(ddqn_metrics.get("shape_group_detail", {})),
            }
            update_metrics_list.append(update_metrics)

        update_elapsed = time.perf_counter() - update_start_time

        mean_update_metrics = {}
        for key in [
            "td_loss",
            "grad_norm",
            "q_mean",
            "q_std",
            "target_mean",
            "td_error_mean",
            "td_error_std",
            "reward_mean",
            "batch_cpu_mb",
            "batch_gpu_mb",
            "archive_rows_mean",
            "candidate_rows_mean",
        ]:
            mean_update_metrics[key] = float(np.mean([m[key] for m in update_metrics_list]))
        mean_update_metrics["archive_rows_max"] = int(max(m["archive_rows_max"] for m in update_metrics_list))
        mean_update_metrics["candidate_rows_max"] = int(max(m["candidate_rows_max"] for m in update_metrics_list))
        mean_update_metrics["shape_group"] = int(round(np.mean([m["shape_group"] for m in update_metrics_list])))
        group_keys = sorted(
            {
                int(obj)
                for m in update_metrics_list
                for obj in m.get("shape_group_detail", {}).keys()
            }
        )
        mean_update_metrics["shape_group_summary"] = {
            int(obj): float(np.mean([m.get("shape_group_detail", {}).get(int(obj), 0) for m in update_metrics_list]))
            for obj in group_keys
        }
        if bool(getattr(cfg, "cuda_cleanup_after_update", False)):
            _cleanup_cuda_cache(cfg.device, rounds=3, sleep_sec=1.0)
        log(
            f"epoch {epoch}/{logged_epoch_target} | reward/FE = {mean_ep_reward:.4f} | "
            f"replay = {len(replay)} | "
            f"lr = {optimizer.param_groups[0]['lr']:.2e} | "
            f"loss = {mean_update_metrics['td_loss']:.6f} | update_sec = {update_elapsed:.2f}"
        )

        if mean_ep_reward > best_reward:
            log(f"new best {reward_metric_name} at epoch {epoch}: {mean_ep_reward:.4f}")
        best_reward = save_training_checkpoint(
            agent,
            cfg,
            cfg.heldout_problem,
            epoch,
            mean_ep_reward,
            best_reward,
            best_state_dict=pre_update_state_dict,
        )

    if executor is not None:
        executor.shutdown(wait=True)
    if bool(use_ray) and ray_mod is not None and ray_mod.is_initialized():
        ray_mod.shutdown()
    log_fp.close()
    return agent


if __name__ == "__main__":
    args = parse_args()
    train_codein(
        problem_name=args.problem,
        dim=args.dim,
        seed=int(args.seed),
        epoch=args.epoch,
        gamma=args.gamma,
        reward_scheme=int(args.reward_scheme),
        reward_norm=bool(args.reward_norm),
        surrogate_model=str(args.surrogate_model),
        solver=str(args.solver),
        num_workers=args.num_workers,
        eval_batch_size=int(args.batch),
        init_fe=args.init_fe,
        max_fe=args.max_fe,
        updates_per_epoch=args.updates_per_epoch,
        hidden_dim=args.hidden_dim,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        agent_name=str(args.agent_name),
        cuda_cleanup_before_update=args.cuda_cleanup_before_update,
        cuda_cleanup_after_update=args.cuda_cleanup_after_update,
        use_ray=bool(args.ray),
        num_seeds=args.num_seeds,
        temperature=args.temperature,
        value_coef=args.value_coef,
        q_coef=args.q_coef,
    )
