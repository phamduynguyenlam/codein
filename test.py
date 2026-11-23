from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import secrets
import sys
import time
import textwrap
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.nhis import DiscAF, NHIS, tensor_policy_state
from problem.constraints import evaluate_objectives_and_constraints, initial_constraint_scale
from baseline.moead_ego import propose_moead_ego_candidates
from infill import (
    EPDIExploitation,
    EPDIExploration,
    ExpectedHypervolumeImprovement,
    NDA,
    NDPBIConvergence,
    NDPBIDiversity,
    RandomSelection,
    SurrogateParetoImprovement,
    TchebycheffProbabilityOfImprovement,
    USeMOUncertainty,
)
from lhs import latin_hypercube_sample
from solver.moead_ego_solver import run_surrogate_moead_ego
from problem.problem import (
    SUPPORTED_PROBLEMS,
    default_problem_dim,
    get_reference_point,
    normalize_problem_name,
    problem_family,
    reference_front,
    make_problem,
)
from reward import (
    hypervolume,
    pareto_front,
    resolve_default_reward_lambda,
    reward_scheme_1,
    reward_scheme_2,
)
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_gp_surrogates,
    surrogate_model_name,
)


class _NullLogFile:
    def close(self) -> None:
        return None


def make_test_logger(log_path: Path | None):
    if log_path is None:
        def _log(message: str) -> None:
            print(str(message))

        return _log, _NullLogFile()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        log_fp.write(text + "\n")
        log_fp.flush()

    return _log, log_fp


def default_test_log_dir(args: argparse.Namespace) -> Path:
    agent_pth = str(getattr(args, "agent_pth", "") or "").strip().lower()
    if "_rs2_" in agent_pth:
        return Path("testing_logs") / "rs2" / str(args.problem).upper()
    return Path("testing_logs") / str(args.problem).upper()


def default_test_plot_dir(args: argparse.Namespace, *, kind: str) -> Path:
    return Path(str(kind)) / str(args.problem).upper()


def default_test_log_path(args: argparse.Namespace, *, agent_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_name = resolve_compare_infill_name(args)
    compare_algo_name = resolve_compare_algo_name(args)
    compare_parts = []
    if compare_name is not None:
        compare_parts.append(str(compare_name))
    if compare_algo_name is not None:
        compare_parts.append(f"algo_{compare_algo_name}")
    compare_tag = f"_compare_{'_'.join(compare_parts)}" if compare_parts else ""
    primary_tag = resolve_primary_output_tag(args, agent_name=agent_name)
    stem = (
        f"test_{primary_tag}_{str(args.problem).lower()}_"
        f"{str(args.surrogate_model).lower()}_seed{int(args.seed)}"
        f"{compare_tag}_{timestamp}.txt"
    )
    return default_test_log_dir(args) / stem


def resolve_primary_output_tag(args: argparse.Namespace, *, agent_name: str | None = None) -> str:
    primary_policy = resolve_primary_policy_name(args)
    if primary_policy == "nhis":
        solver_name = str(getattr(args, "solver", "")).strip().lower().replace("-", "_")
        solver_family = canonical_solver_family_name(solver_name)
        agent_tag = str(agent_name if agent_name is not None else getattr(args, "agent_name", "nhis")).lower()
        if solver_family is not None:
            return f"{agent_tag}_solver_{solver_family}"
        return agent_tag
    return str(primary_policy).lower()


def _normalize_display_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace("/", "_")


def resolve_compare_infill_name(args: argparse.Namespace) -> str | None:
    names = resolve_compare_infill_names(args)
    if len(names) <= 0:
        return None
    return ",".join(names)


def resolve_compare_algo_name(args: argparse.Namespace) -> str | None:
    names = resolve_compare_algo_names(args)
    if len(names) <= 0:
        return None
    return ",".join(names)


def resolve_compare_infill_names(args: argparse.Namespace) -> list[str]:
    raw_value = getattr(args, "compare_infill", None)
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip().lower()
    if raw_text == "":
        return []
    all_names = [
        "ehvi",
        "usemo_uncertainty",
        "spi",
        "pi",
        "random",
        "nd_a",
        "nd_pbi_convergence",
        "nd_pbi_diversity",
        "epdi_exploitation",
        "epdi_exploration",
    ]
    allowed = set(all_names + ["all"])
    tokens = [part.strip().lower().replace("-", "_") for part in raw_text.split(",")]
    tokens = [token for token in tokens if token != ""]
    if len(tokens) <= 0:
        return []

    resolved: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(
                f"Unsupported compare_infill entry: {token}. "
                "Expected entries from infill.py or all."
            )
        if token == "all":
            for name in all_names:
                if name not in resolved:
                    resolved.append(name)
            continue
        if token not in resolved:
            resolved.append(token)
    return resolved


def resolve_compare_algo_names(args: argparse.Namespace) -> list[str]:
    raw_value = getattr(args, "compare_algo", None)
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip().lower()
    if raw_text == "":
        return []
    allowed = {
        "nhis",
        "disc_af",
        "moead_ego",
        "all",
    }
    tokens = [part.strip().lower().replace("-", "_") for part in raw_text.split(",")]
    tokens = [token for token in tokens if token != ""]
    if len(tokens) <= 0:
        return []
    resolved: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(
                f"Unsupported compare_algo entry: {token}. "
                "Expected entries from {'nhis','disc_af','moead_ego','all'}."
            )
        if token == "all":
            for name in ["moead_ego"]:
                if name not in resolved:
                    resolved.append(name)
            continue
        if token not in resolved:
            resolved.append(token)
    if "nhis" in resolved:
        resolved = ["nhis"] + [token for token in resolved if token != "nhis"]
    return resolved


def resolve_infill_name(args: argparse.Namespace) -> str | None:
    raw_value = getattr(args, "infill", None)
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text == "":
        return None
    return text.replace("-", "_")


def resolve_primary_policy_name(args: argparse.Namespace) -> str:
    infill_name = resolve_infill_name(args)
    agent_name = str(getattr(args, "agent_name", "nhis")).strip().lower().replace("-", "_")
    if infill_name is None:
        if agent_name == "disc_af":
            return agent_name
        return "nhis"
    if infill_name == "nhis":
        return "nhis"
    if infill_name in {"disc_af", "af"}:
        return "disc_af"
    return str(infill_name)


def primary_policy_display_name(name: str) -> str:
    key = _normalize_display_key(name)
    if key == "nhis":
        return "NHIS"
    if key == "disc_af":
        return "DISC w/o candidate pool interaction"
    return compare_infill_display_name(key)


def is_agent_policy(name: str) -> bool:
    return str(name).strip().lower().replace("-", "_") in {
        "nhis",
        "disc_af",
    }


def compare_infill_display_name(name: str) -> str:
    display_map = {
        "ehvi": "EHVI",
        "usemo_uncertainty": "USeMO-Uncertainty",
        "spi": "SPI",
        "surrogate_pareto_improvement": "SPI",
        "pi": "PI",
        "tchebycheff_probability_of_improvement": "PI",
        "random": "RANDOM",
        "nd_a": "ND-A",
        "nd_pbi_convergence": "ND-PBI-Convergence",
        "nd_pbi_diversity": "ND-PBI-Diversity",
        "epdi_exploitation": "EPDI-Exploitation",
        "epdi_exploration": "EPDI-Exploration",
        "moea_d_ego": "MOEA-D/EGO",
        "moead_ego": "MOEA-D/EGO",
    }
    key = _normalize_display_key(name)
    return display_map.get(key, key.upper())


def compare_algo_display_name(name: str) -> str:
    display_map = {
        "nhis": "NHIS",
        "disc_af": "DISC w/o candidate pool interaction",
        "nsga2": "NSGA-II",
        "moead_ego": "MOEA-D/EGO",
        "moea_d_ego": "MOEA-D/EGO",
    }
    key = _normalize_display_key(name)
    return display_map.get(key, key.upper())


def resolve_framework_label(args: argparse.Namespace) -> str:
    return "NHIS"


def resolve_infill_label(args: argparse.Namespace) -> str:
    primary_policy = resolve_primary_policy_name(args)
    return primary_policy_display_name(primary_policy)


def canonical_solver_family_name(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower().replace("-", "_").replace("/", "_")
    if key in {"moead_ego", "moea_d_ego"}:
        return "moead_ego"
    return None


def resolve_solver_baseline_metadata(args: argparse.Namespace) -> tuple[str | None, str | None]:
    solver_name = str(getattr(args, "solver", "")).strip().lower().replace("-", "_")
    solver_family = canonical_solver_family_name(solver_name)
    if solver_family is not None:
        return "solver_vs_baseline", "nhis_solver"
    return None, None


def build_compare_infill_criterion(name: str, *, ref_point: np.ndarray):
    key = str(name).strip().lower().replace("-", "_")
    if key == "ehvi":
        return ExpectedHypervolumeImprovement(ref_point=ref_point, n_samples=128)
    if key == "usemo_uncertainty":
        return USeMOUncertainty()
    if key in {"spi", "surrogate_pareto_improvement"}:
        return SurrogateParetoImprovement()
    if key in {"pi", "tchebycheff_probability_of_improvement"}:
        return TchebycheffProbabilityOfImprovement()
    if key == "random":
        return RandomSelection()
    if key == "nd_a":
        return NDA()
    if key == "nd_pbi_convergence":
        return NDPBIConvergence()
    if key == "nd_pbi_diversity":
        return NDPBIDiversity()
    if key == "epdi_exploitation":
        return EPDIExploitation()
    if key == "epdi_exploration":
        return EPDIExploration()
    raise ValueError(f"Unsupported compare_infill: {name}")


def _call_infill_select_index(criterion, **kwargs) -> tuple[int, np.ndarray]:
    signature = inspect.signature(criterion.select_index)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        filtered_kwargs = kwargs
    else:
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return criterion.select_index(**filtered_kwargs)


def _make_infill_selection_kwargs(
    *,
    criterion,
    archive_y: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    seed: int,
    offspring_info: dict[str, Any] | None = None,
    progress_ratio: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "archive_y": archive_y,
        "candidate_mean": candidate_mean,
        "candidate_std": candidate_std,
        "seed": int(seed),
    }
    if offspring_info is not None:
        candidate_weight_vectors = offspring_info.get("candidate_weight_vectors")
        if candidate_weight_vectors is not None:
            kwargs["candidate_weight_vectors"] = np.asarray(
                candidate_weight_vectors,
                dtype=np.float32,
            )
    del criterion, progress_ratio
    return kwargs


def resolve_test_reward_scheme(args: argparse.Namespace) -> int:
    agent_pth = getattr(args, "agent_pth", None)
    if not agent_pth:
        return 1
    match = re.search(r"rs(\d+)", Path(str(agent_pth)).name.lower())
    if match is None:
        return 1
    reward_scheme_id = int(match.group(1))
    if reward_scheme_id not in {1, 2}:
        raise ValueError(
            f"Unsupported reward scheme tag rs{reward_scheme_id} in checkpoint filename; "
            "expected rs1 or rs2."
        )
    return reward_scheme_id


def is_random_agent_path(agent_pth: str | None) -> bool:
    return agent_pth is not None and str(agent_pth).strip().lower() in {"random", "random_model"}


def resolve_default_best_reward_checkpoint(args: argparse.Namespace) -> str:
    policy_name = resolve_primary_policy_name(args)
    checkpoint_root = Path("training_logs") / policy_name
    filename_pattern = (
        f"{policy_name}_problem_{problem_family(args.problem).lower()}_*_best_reward.pth"
    )
    candidates = [
        path
        for path in checkpoint_root.rglob(filename_pattern)
        if path.is_file()
    ] if checkpoint_root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No best-reward checkpoint found for {policy_name} on {args.problem} "
            f"under {checkpoint_root}."
        )
    rs1_candidates = [path for path in candidates if "_rs1_" in path.name.lower()]
    if rs1_candidates:
        candidates = rs1_candidates
    selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return str(selected.resolve())


def resolve_inference_source(
    *,
    policy_name: str,
    agent_pth: str | None = None,
    random_model: bool = False,
    compare_algo: str | None = None,
) -> str:
    key = str(policy_name).strip().lower().replace("-", "_")
    if compare_algo is not None:
        algo_name = str(compare_algo).strip().lower().replace("-", "_")
        if bool(random_model):
            return f"{algo_name}:random_model"
        if agent_pth:
            return str(Path(str(agent_pth)).resolve())
        return f"{algo_name}:uninitialized_model"
    if is_agent_policy(key):
        if bool(random_model):
            return f"{key}:random_model"
        if agent_pth:
            return str(Path(str(agent_pth)).resolve())
        return f"{key}:uninitialized_model"
    return f"infill:{key}"


def format_policy_prefix(policy_name: str) -> str:
    return f"[{str(policy_name).strip().lower().replace('-', '_')}] "


def compute_test_reward(
    *,
    reward_scheme_id: int,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float,
    true_pareto_hv: float | None = None,
) -> float:
    if int(reward_scheme_id) == 1:
        return float(
            reward_scheme_1(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                reward_lambda=float(reward_lambda),
            )
        )
    if int(reward_scheme_id) == 2:
        if true_pareto_hv is None:
            raise ValueError("reward_scheme_2 requires true_pareto_hv in tester.")
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
                reward_lambda=float(reward_lambda),
            )
        )
    raise ValueError(f"Unsupported reward_scheme_id for tester: {reward_scheme_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NHIS-guided surrogate-assisted optimization with 80 LHS init + 40 evolution steps."
    )
    parser.add_argument("--problem", type=normalize_problem_name, default="MW1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=None)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", default="auto")
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=None)
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logit_scale", type=float, default=5.0)
    parser.add_argument("--agent_pth", type=str, default=None)
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="q_greedy",
        choices=["epsilon_greedy", "q_greedy", "softmax_sample"],
        help="Decode agent Q-values with epsilon-greedy, greedy, or softmax sampling.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--infill", type=str, default=None)
    parser.add_argument("--compare_infill", type=str, default=None)
    parser.add_argument("--compare_algo", type=str, default=None)
    parser.add_argument("--compare_agent_pth", type=str, default=None)
    parser.add_argument("--solver", default="moead_ego", choices=["moead_ego"])
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    args.surrogate_model = "gp"
    args.reward_lambda = None
    args.batch = 1
    args.problem = str(args.problem).upper()
    args.offspring_size_user = args.offspring_size is not None
    args.saea_steps = 50
    if not args.offspring_size_user:
        args.offspring_size = 150
    if is_random_agent_path(args.compare_agent_pth):
        args.compare_agent_pth = None
        args.compare_random_model = True
    else:
        args.compare_random_model = False
    if args.dim is None:
        args.dim = default_problem_dim(args.problem)

    if args.init_fe < 2:
        parser.error("--init_fe must be at least 2.")
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds)
                                   or any(seed < 0 or seed >= 2**32 for seed in args.seeds)):
        parser.error("--seeds must contain distinct integers between 0 and 2**32 - 1.")
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    if int(args.offspring_size) < 2:
        raise ValueError(f"offspring_size must be at least 2, got {args.offspring_size}.")
    if args.num_seeds < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("num_seeds must be positive and temperature must be finite and positive.")
    return args


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_seed(seed_value: Any) -> int:
    text = str(seed_value).strip().lower()
    if text in {"", "auto", "none"}:
        return int((time.time_ns() ^ os.getpid() ^ secrets.randbits(32)) % (2**31 - 1))
    return int(seed_value)


def fe_progress(*, init_fe: int, step: int, max_fe: int) -> float:
    max_fe_value = int(max_fe)
    if max_fe_value <= 0:
        raise ValueError(f"max_fe must be positive, got {max_fe_value}.")
    current_fe = min(int(init_fe) + int(step), max_fe_value - 1)
    return float(current_fe) / float(max_fe_value)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def resolve_agent_cls(agent_name: str):
    name = str(agent_name).strip().lower()
    if name == "nhis":
        return NHIS
    if name == "disc_af":
        return DiscAF
    raise ValueError(f"Unsupported agent_name: {agent_name}")


def build_named_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray, name: str):
    surrogate_name = str(name).lower()
    if surrogate_name == "gp":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            nu=float(getattr(args, "gp_nu", 5.0)),
        )

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def build_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray):
    return build_named_surrogate(args, archive_x, archive_y, surrogate_model_name(args))


def make_nsga2_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def generate_offspring_pool(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    step: int,
    seed_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    solver_name = str(getattr(args, "solver", "moead_ego")).lower()
    if solver_name != "moead_ego":
        raise ValueError(f"Unsupported solver: {solver_name}")
    result = run_surrogate_moead_ego(
        problem=nsga_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate=build_surrogate(args, archive_x, archive_y),
        pop_size=int(args.offspring_size) if bool(getattr(args, "offspring_size_user", False)) else None,
        saea_steps=int(args.saea_steps),
        seed=int(args.seed) + int(step),
        seed_x=seed_x,
    )
    return (
        np.asarray(result.x, dtype=np.float32),
        np.asarray(result.mean, dtype=np.float32),
        np.asarray(result.std, dtype=np.float32),
        [np.arange(int(result.x.shape[0]), dtype=np.int64)],
        {
            "candidate_weight_vectors": np.asarray(result.weight_vectors, dtype=np.float32),
            "eti": np.asarray(result.eti, dtype=np.float32),
        },
    )


def refresh_offspring_predictions(
    *,
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offspring_x_arr = np.asarray(offspring_x, dtype=np.float32)
    if int(offspring_x_arr.shape[0]) <= 0:
        n_obj = int(np.asarray(archive_y, dtype=np.float32).shape[1])
        empty = np.empty((0, n_obj), dtype=np.float32)
        return empty, empty.copy()
    surrogate = build_surrogate(
        args,
        np.asarray(archive_x, dtype=np.float32),
        np.asarray(archive_y, dtype=np.float32),
    )
    refreshed_mean = predict_surrogate_mean(surrogate, offspring_x_arr)
    refreshed_std = predict_surrogate_std(surrogate, offspring_x_arr)
    return (
        np.asarray(refreshed_mean, dtype=np.float32),
        np.asarray(refreshed_std, dtype=np.float32),
    )


def predict_surrogate_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(
    surrogate: Any,
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    surrogate: Any,
) -> np.ndarray:
    archive_y = np.asarray(archive_y, dtype=np.float32)

    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)

    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=archive_x,
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=offspring_x,
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def build_agent(
    args: argparse.Namespace,
    *,
    map_location: str,
    agent_name: str = "nhis",
    state_dict_override: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, dict[str, float]]:
    agent_cls = resolve_agent_cls(agent_name)
    torch_load_sec = 0.0
    load_state_dict_sec = 0.0
    state_dict = None if state_dict_override is None else state_dict_override

    if state_dict is None and args.agent_pth and not bool(args.random_model):
        torch_load_started_at = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            state = torch.load(args.agent_pth, map_location=map_location)
        torch_load_sec = time.perf_counter() - torch_load_started_at
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    agent_kwargs = {
        "hidden_dim": int(args.hidden_dim),
        "n_heads": int(args.n_heads),
        "ff_dim": int(args.ff_dim),
        "dropout": float(args.dropout),
    }
    if agent_cls is not NHIS:
        agent_kwargs["logit_scale"] = float(args.logit_scale)
    model_to_device_started_at = time.perf_counter()
    disc = agent_cls(**agent_kwargs).to(map_location)
    model_to_device_sec = time.perf_counter() - model_to_device_started_at
    disc.eval()

    if state_dict is not None:
        load_state_dict_started_at = time.perf_counter()
        disc.load_state_dict(state_dict, strict=True)
        load_state_dict_sec = time.perf_counter() - load_state_dict_started_at

    return disc, {
        "model_to_device_sec": float(model_to_device_sec),
        "torch_load_sec": float(torch_load_sec),
        "load_state_dict_sec": float(load_state_dict_sec),
    }


def prediction_history_kwargs(agent, y_true_history, sigma_true_history):
    if not bool(getattr(agent, "uses_prediction_history", False)):
        return {}
    return {
        "y_true_history": y_true_history,
        "sigma_true_history": sigma_true_history,
    }


def build_prediction_history_kwargs(agent, y_history: np.ndarray, sigma_history: np.ndarray, *, device: str):
    if not bool(getattr(agent, "uses_prediction_history", False)):
        return {}
    return {
        "y_true_history": torch.from_numpy(np.asarray(y_history, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        ),
        "sigma_true_history": torch.from_numpy(np.asarray(sigma_history, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        ),
    }


def build_trajectory_history_kwargs(
    agent,
    state_embeddings: list[torch.Tensor],
    timesteps: list[float],
    q_values: list[float],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    if not bool(getattr(agent, "uses_trajectory_history", False)) or not state_embeddings:
        return {}
    embeddings = torch.stack(state_embeddings, dim=0).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    timestep_tensor = torch.tensor(timesteps, device=device, dtype=torch.float32).unsqueeze(0)
    q_tensor = torch.tensor(q_values, device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(state_embeddings), device=device, dtype=torch.bool)
    return {
        "history_state_embeddings": embeddings,
        "history_timesteps": timestep_tensor,
        "history_q_values": q_tensor,
        "history_mask": mask,
    }


def append_trajectory_history(
    agent,
    output: dict[str, torch.Tensor],
    selected_idx: int,
    progress: float,
    state_embeddings: list[torch.Tensor],
    timesteps: list[float],
    q_values: list[float],
) -> None:
    if not bool(getattr(agent, "uses_trajectory_history", False)):
        return
    embeddings = output.get("state_action_embeddings")
    scores = output.get("q_values")
    if embeddings is None or scores is None:
        raise RuntimeError("Trajectory-aware agent must return state_action_embeddings and q_values.")
    state_embeddings.append(embeddings[0, int(selected_idx)].detach().cpu())
    timesteps.append(float(progress))
    q_values.append(float(scores[0, int(selected_idx)].detach().cpu().item()))


def init_archive_prediction_history(archive_y: np.ndarray, *, history_steps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    y_history = np.repeat(archive_y_arr[:, None, :], int(history_steps), axis=1).astype(np.float32)
    sigma_history = np.zeros_like(y_history, dtype=np.float32)
    return y_history, sigma_history


def append_archive_prediction_history(
    y_history: np.ndarray,
    sigma_history: np.ndarray,
    *,
    selected_pred: np.ndarray,
    selected_sigma: np.ndarray,
    selected_true: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_steps = int(np.asarray(y_history).shape[1])
    pred = np.asarray(selected_pred, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(selected_sigma, dtype=np.float32).reshape(1, -1)
    true = np.asarray(selected_true, dtype=np.float32).reshape(1, -1)
    new_y_history = np.repeat(pred[:, None, :], history_steps, axis=1).astype(np.float32)
    new_sigma_history = np.repeat(sigma[:, None, :], history_steps, axis=1).astype(np.float32)
    eval_step = min(max(int(step), 0), history_steps - 1)
    new_y_history[:, eval_step:, :] = true[:, None, :]
    new_sigma_history[:, eval_step:, :] = 0.0
    return (
        np.vstack([np.asarray(y_history, dtype=np.float32), new_y_history]).astype(np.float32),
        np.vstack([np.asarray(sigma_history, dtype=np.float32), new_sigma_history]).astype(np.float32),
    )


@dataclass
class StepRecord:
    step: int
    fe: int
    selected_index: int
    selected_x: list[float]
    surrogate_y: list[float]
    true_y: list[float]
    reward: float
    hv: float
    igd: float | None
    archive_size: int


def feasible_metrics(archive_y, archive_g, ref_point, true_pareto):
    feasible = archive_y[(archive_g <= 0).all(axis=1)]
    if not len(feasible):
        return feasible, 0.0, None
    front = pareto_front(feasible)
    # Distances run from each true-front point to its nearest approximation.
    igd = None if true_pareto is None else float(cKDTree(front).query(true_pareto)[0].mean())
    return front, hypervolume(front, ref_point), igd


def format_igd(igd):
    return "NA" if igd is None else f"{igd:.12f}"


def remove_offspring_candidate(
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    offspring_groups: list[np.ndarray],
    selected_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    n_candidates = int(np.asarray(offspring_x).shape[0])
    keep_mask = np.ones(n_candidates, dtype=bool)
    keep_mask[int(selected_idx)] = False
    keep_idx = np.where(keep_mask)[0].astype(np.int64)
    old_to_new = np.full(n_candidates, -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(int(keep_idx.shape[0]), dtype=np.int64)

    new_groups: list[np.ndarray] = []
    for group in offspring_groups:
        group_arr = np.asarray(group, dtype=np.int64).reshape(-1)
        group_arr = group_arr[group_arr != int(selected_idx)]
        remapped = old_to_new[group_arr]
        remapped = remapped[remapped >= 0]
        if remapped.size > 0:
            new_groups.append(remapped.astype(np.int64))
    if len(new_groups) == 0 and keep_idx.size > 0:
        new_groups = [np.arange(int(keep_idx.shape[0]), dtype=np.int64)]

    return (
        np.asarray(offspring_x, dtype=np.float32)[keep_idx],
        np.asarray(offspring_pred, dtype=np.float32)[keep_idx],
        np.asarray(offspring_sigma, dtype=np.float32)[keep_idx],
        new_groups,
    )


def remove_offspring_info_candidate(
    offspring_info: dict[str, Any],
    *,
    selected_idx: int,
    n_candidates: int,
) -> dict[str, Any]:
    """Keep candidate-aligned solver metadata synchronized with a shrinking pool."""
    updated = dict(offspring_info)
    for key in ("candidate_weight_vectors", "eti"):
        value = updated.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and int(array.shape[0]) == int(n_candidates):
            updated[key] = np.delete(array, int(selected_idx), axis=0)
    return updated


def run_policy_rollout_batched(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    archive_x_init: np.ndarray,
    archive_y_init: np.ndarray,
    archive_g_init: np.ndarray,
    policy_name: str,
    disc: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    make_plot: bool = True,
    logger=print,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_x = np.asarray(archive_x_init, dtype=np.float32).copy()
    archive_y = np.asarray(archive_y_init, dtype=np.float32).copy()
    archive_y_history, archive_sigma_history = init_archive_prediction_history(archive_y, history_steps=40)
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    batch_size = max(1, int(getattr(args, "batch", 1)))
    fe_history = [int(args.init_fe)]
    archive_g = np.asarray(archive_g_init, dtype=np.float32).copy()
    front, hv, igd = feasible_metrics(archive_y, archive_g, ref_point, true_pareto)
    hv_history = [hv]
    igd_history = [igd]
    history: list[StepRecord] = []
    trajectory_state_embeddings: list[torch.Tensor] = []
    trajectory_timesteps: list[float] = []
    trajectory_q_values: list[float] = []
    step_rewards: list[float] = []
    if infill_criterion is not None and hasattr(infill_criterion, "reset"):
        infill_criterion.reset()

    prefix = format_policy_prefix(policy_name)
    logger(f"{prefix}iter 0 | FE = {fe_history[0]} | front = {len(front)} | HV = {hv:.12f} | IGD = {format_igd(igd)}")

    step = 0
    pool_id = 0
    policy_key = str(policy_name).lower()
    nhis = policy_key == "nhis"
    if nhis:
        if disc is None:
            raise ValueError("NHIS rollout requires a built NHIS agent.")
        if args.policy_mode not in {"q_greedy", "softmax_sample"}:
            raise ValueError("NHIS inference supports q_greedy or softmax_sample.")
        scale = initial_constraint_scale(archive_g)
        bounds = {
            "lower_bound": np.full(int(args.dim), float(problem.lower), dtype=np.float32),
            "upper_bound": np.full(int(args.dim), float(problem.upper), dtype=np.float32),
        }
    while step < n_evo_steps:
        seed_x = None
        if nhis:
            population_size = args.offspring_size if args.offspring_size_user else len(archive_x)
            seed_count = min(int(args.num_seeds), len(archive_x), int(population_size))
            archive_state = dict(
                x_true=archive_x, y_true=archive_y, g_true=archive_g, constraint_scale=scale,
                seed_count=seed_count, progress=(int(args.init_fe) + step) / int(args.max_fe), **bounds,
            )
            with torch.no_grad():
                indices, _, _, _ = disc.choose_seeds(
                    sample=False, **tensor_policy_state(archive_state, args.device),
                )
            seed_x = archive_x[indices[0].cpu().numpy()]
        offspring_x, offspring_pred, offspring_sigma, offspring_groups, offspring_info = generate_offspring_pool(
            args=args,
            nsga_problem=nsga2_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            step=pool_id,
            seed_x=seed_x,
        )
        if nhis:
            if archive_g.shape[1]:
                constraint_gp = build_surrogate(args, archive_x, archive_g)
                g_sur_mean = constraint_gp.predict_mean(offspring_x)
                g_sur_std = constraint_gp.predict_std(offspring_x)
            else:
                g_sur_mean = np.empty((len(offspring_x), 0), dtype=np.float32)
                g_sur_std = g_sur_mean.copy()
        pool_id += 1
        pool_evals = min(batch_size, n_evo_steps - step)

        for _ in range(pool_evals):
            if step >= n_evo_steps or int(offspring_x.shape[0]) <= 0:
                break
            if nhis:
                state = dict(
                    x_true=archive_x, y_true=archive_y, g_true=archive_g, constraint_scale=scale,
                    seed_count=seed_count, progress=(int(args.init_fe) + step) / int(args.max_fe), **bounds,
                    x_sur=offspring_x, y_sur=offspring_pred, sigma_sur=offspring_sigma,
                    g_sur_mean=g_sur_mean, g_sur_std=g_sur_std,
                )
                with torch.no_grad():
                    out = disc(**tensor_policy_state(state, args.device, include_offspring=True))
                    selected_idx = int(disc.choose_candidate(
                        out, sample=args.policy_mode == "softmax_sample", temperature=float(args.temperature)
                    ).item())
            elif policy_key == "disc_af":
                if disc is None:
                    raise ValueError("DISC rollout requires a built disc model.")
                progress = fe_progress(
                    init_fe=int(args.init_fe),
                    step=int(step),
                    max_fe=int(args.max_fe),
                )
                history_kwargs = build_prediction_history_kwargs(
                    disc,
                    archive_y_history,
                    archive_sigma_history,
                    device=str(args.device),
                )
                history_kwargs.update(
                    build_trajectory_history_kwargs(
                        disc,
                        trajectory_state_embeddings,
                        trajectory_timesteps,
                        trajectory_q_values,
                        device=str(args.device),
                    )
                )
                with torch.no_grad():
                    out = disc(
                        x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                        y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                        x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                        y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                        sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                        progress=progress,
                        lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                        upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                        decode_type=str(args.policy_mode),
                        epsilon=0.05,
                        **history_kwargs,
                    )
                selected_idx = int(out["ranking"].reshape(out["ranking"].shape[0], -1)[0, 0].item())
                append_trajectory_history(
                    disc,
                    out,
                    selected_idx,
                    progress,
                    trajectory_state_embeddings,
                    trajectory_timesteps,
                    trajectory_q_values,
                )
            elif infill_criterion is not None:
                selected_idx, _ = _call_infill_select_index(
                    infill_criterion,
                    **_make_infill_selection_kwargs(
                        criterion=infill_criterion,
                        archive_y=archive_y,
                        candidate_mean=offspring_pred,
                        candidate_std=offspring_sigma,
                        seed=int(args.seed) + step,
                        offspring_info=offspring_info,
                        progress_ratio=fe_progress(
                            init_fe=int(args.init_fe),
                            step=int(step),
                            max_fe=int(args.max_fe),
                        ),
                    ),
                )
            else:
                raise ValueError(f"Unsupported batched policy_name: {policy_name}")

            previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
            selected_x = offspring_x[selected_idx : selected_idx + 1]
            selected_pred = offspring_pred[selected_idx]
            selected_sigma = offspring_sigma[selected_idx]
            selected_true, selected_g = evaluate_objectives_and_constraints(problem, selected_x)
            archive_g = np.vstack((archive_g, selected_g))
            if nhis:
                g_sur_mean = np.delete(g_sur_mean, selected_idx, axis=0)
                g_sur_std = np.delete(g_sur_std, selected_idx, axis=0)
            step_reward = compute_test_reward(
                reward_scheme_id=int(reward_scheme_id),
                previous_front=previous_front,
                selected_objectives=selected_true,
                ref_point=ref_point,
                reward_lambda=float(args.reward_lambda),
                true_pareto_hv=true_pareto_hv,
            )

            archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
            archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
            archive_y_history, archive_sigma_history = append_archive_prediction_history(
                archive_y_history,
                archive_sigma_history,
                selected_pred=selected_pred,
                selected_sigma=selected_sigma,
                selected_true=selected_true,
                step=step,
            )
            step += 1
            front, hv, igd = feasible_metrics(archive_y, archive_g, ref_point, true_pareto)
            fe = int(args.init_fe) + step
            front_size = len(front)
            fe_history.append(fe)
            hv_history.append(float(hv))
            igd_history.append(igd)

            record = StepRecord(
                step=step,
                fe=fe,
                selected_index=int(selected_idx),
                selected_x=selected_x.reshape(-1).astype(float).tolist(),
                surrogate_y=selected_pred.astype(float).tolist(),
                true_y=selected_true.reshape(-1).astype(float).tolist(),
                reward=step_reward,
                hv=float(hv),
                igd=igd,
                archive_size=int(archive_x.shape[0]),
            )
            history.append(record)
            step_rewards.append(step_reward)

            log_line = f"{prefix}iter {record.step} | FE = {record.fe} | front = {front_size} | HV = {record.hv:.12f} | IGD = {format_igd(record.igd)}"
            log_line += f" | reward = {record.reward:.6f}"
            logger(log_line)

            old_pool_size = int(offspring_x.shape[0])
            offspring_x, offspring_pred, offspring_sigma, offspring_groups = remove_offspring_candidate(
                offspring_x,
                offspring_pred,
                offspring_sigma,
                offspring_groups,
                selected_idx=int(selected_idx),
            )
            offspring_info = remove_offspring_info_candidate(
                offspring_info,
                selected_idx=int(selected_idx),
                n_candidates=old_pool_size,
            )
            if (
                policy_key == "disc_af"
                and int(offspring_x.shape[0]) > 0
            ):
                offspring_pred, offspring_sigma = refresh_offspring_predictions(
                    args=args,
                    archive_x=archive_x,
                    archive_y=archive_y,
                    offspring_x=offspring_x,
                )

    final_front = front
    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "batch": batch_size,
        "surrogate_model": surrogate_model_name(args),
        "candidate_solver": str(getattr(args, "solver", "moead_ego")).lower(),
        "reward_lambda": float(args.reward_lambda),
        "reward_scheme": int(reward_scheme_id),
        "agent_name": args.agent_name,
        "policy_name": policy_key,
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": hv_history[-1],
        "final_igd": igd_history[-1],
        "archive_x": archive_x.tolist(),
        "archive_f": archive_y.tolist(),
        "archive_g": archive_g.tolist(),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "history": [asdict(item) for item in history],
        "fe_history": fe_history,
        "hv_history": hv_history,
        "igd_history": igd_history,
    }
    if make_plot:
        curve = aggregate_metric_histories([summary])
        summary["plot_path"], _ = plot_metric_curves(
            args, [(primary_policy_display_name(policy_name), curve, summary)],
            true_pareto, f"seed{args.seed}",
        )
    return summary, archive_y


def run_policy_rollout(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    archive_x_init: np.ndarray,
    archive_y_init: np.ndarray,
    archive_g_init: np.ndarray,
    policy_name: str,
    disc: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    make_plot: bool = True,
    logger=print,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    batch_size = max(1, int(getattr(args, "batch", 1)))
    policy_key = str(policy_name).lower()
    if policy_key == "nhis" or (batch_size > 1 and policy_key != "moead_ego"):
        return run_policy_rollout_batched(
            args=args,
            problem=problem,
            nsga2_problem=nsga2_problem,
            ref_point=ref_point,
            true_pareto=true_pareto,
            archive_x_init=archive_x_init,
            archive_y_init=archive_y_init,
            archive_g_init=archive_g_init,
            policy_name=policy_name,
            disc=disc,
            infill_criterion=infill_criterion,
            compare_mode=compare_mode,
            make_plot=make_plot,
            logger=logger,
            reward_scheme_id=reward_scheme_id,
            true_pareto_hv=true_pareto_hv,
        )

    archive_x = np.asarray(archive_x_init, dtype=np.float32).copy()
    archive_y = np.asarray(archive_y_init, dtype=np.float32).copy()
    archive_y_history, archive_sigma_history = init_archive_prediction_history(archive_y, history_steps=40)
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    fe_history = [int(args.init_fe)]
    archive_g = np.asarray(archive_g_init, dtype=np.float32).copy()
    front, hv, igd = feasible_metrics(archive_y, archive_g, ref_point, true_pareto)
    hv_history = [hv]
    igd_history = [igd]
    history: list[StepRecord] = []
    trajectory_state_embeddings: list[torch.Tensor] = []
    trajectory_timesteps: list[float] = []
    trajectory_q_values: list[float] = []
    step_rewards: list[float] = []
    if infill_criterion is not None and hasattr(infill_criterion, "reset"):
        infill_criterion.reset()

    prefix = format_policy_prefix(policy_name)
    logger(f"{prefix}iter 0 | FE = {fe_history[0]} | front = {len(front)} | HV = {hv:.12f} | IGD = {format_igd(igd)}")

    for step in range(n_evo_steps):
        policy_key = policy_name.lower()
        offspring_info: dict[str, Any] = {}
        if policy_key == "moead_ego":
            offspring_x, offspring_pred, offspring_sigma, moead_info = propose_moead_ego_candidates(
                problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                pop_size=int(args.offspring_size) if bool(getattr(args, "offspring_size_user", False)) else None,
                moead_gen=int(args.saea_steps),
                batch_size=1,
                gp_restarts=3,
                seed=int(args.seed) + int(step),
            )
            offspring_groups = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
            offspring_info = {
                "reference_vectors": None if moead_info is None else moead_info.get("reference_vectors")
            }
        else:
            offspring_x, offspring_pred, offspring_sigma, offspring_groups, offspring_info = generate_offspring_pool(
                args=args,
                nsga_problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                step=step,
            )
        if policy_key == "disc_af":
            if disc is None:
                raise ValueError("DISC rollout requires a built disc model.")
            progress = fe_progress(
                init_fe=int(args.init_fe),
                step=int(step),
                max_fe=int(args.max_fe),
            )
            history_kwargs = build_prediction_history_kwargs(
                disc,
                archive_y_history,
                archive_sigma_history,
                device=str(args.device),
            )
            history_kwargs.update(
                build_trajectory_history_kwargs(
                    disc,
                    trajectory_state_embeddings,
                    trajectory_timesteps,
                    trajectory_q_values,
                    device=str(args.device),
                )
            )
            with torch.no_grad():
                out = disc(
                    x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                    y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                    x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                    y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                    sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                    progress=progress,
                    lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                    upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                    decode_type=str(args.policy_mode),
                    epsilon=0.05,
                    **history_kwargs,
                )
            selected_idx = int(out["ranking"].reshape(out["ranking"].shape[0], -1)[0, 0].item())
            append_trajectory_history(
                disc,
                out,
                selected_idx,
                progress,
                trajectory_state_embeddings,
                trajectory_timesteps,
                trajectory_q_values,
            )
        elif policy_name.lower() == "moead_ego":
            selected_idx = 0
        elif infill_criterion is not None:
            selected_idx, _ = _call_infill_select_index(
                infill_criterion,
                **_make_infill_selection_kwargs(
                    criterion=infill_criterion,
                    archive_y=archive_y,
                    candidate_mean=offspring_pred,
                    candidate_std=offspring_sigma,
                    seed=int(args.seed) + step,
                    offspring_info=offspring_info,
                    progress_ratio=fe_progress(
                        init_fe=int(args.init_fe),
                        step=int(step),
                        max_fe=int(args.max_fe),
                    ),
                ),
            )
        else:
            raise ValueError(f"Unsupported policy_name: {policy_name}")

        previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        selected_x = offspring_x[selected_idx : selected_idx + 1]
        selected_pred = offspring_pred[selected_idx]
        selected_sigma = offspring_sigma[selected_idx]
        selected_true, selected_g = evaluate_objectives_and_constraints(problem, selected_x)
        archive_g = np.vstack((archive_g, selected_g))
        step_reward = compute_test_reward(
            reward_scheme_id=int(reward_scheme_id),
            previous_front=previous_front,
            selected_objectives=selected_true,
            ref_point=ref_point,
            reward_lambda=float(args.reward_lambda),
            true_pareto_hv=true_pareto_hv,
        )

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
        archive_y_history, archive_sigma_history = append_archive_prediction_history(
            archive_y_history,
            archive_sigma_history,
            selected_pred=selected_pred,
            selected_sigma=selected_sigma,
            selected_true=selected_true,
            step=step,
        )
        front, hv, igd = feasible_metrics(archive_y, archive_g, ref_point, true_pareto)
        fe = int(args.init_fe) + step + 1
        front_size = len(front)
        fe_history.append(fe)
        hv_history.append(float(hv))
        igd_history.append(igd)

        record = StepRecord(
            step=step + 1,
            fe=fe,
            selected_index=selected_idx,
            selected_x=selected_x.reshape(-1).astype(float).tolist(),
            surrogate_y=selected_pred.astype(float).tolist(),
            true_y=selected_true.reshape(-1).astype(float).tolist(),
            reward=step_reward,
            hv=float(hv),
            igd=igd,
            archive_size=int(archive_x.shape[0]),
        )
        history.append(record)
        step_rewards.append(step_reward)

        log_line = f"{prefix}iter {record.step} | FE = {record.fe} | front = {front_size} | HV = {record.hv:.12f} | IGD = {format_igd(record.igd)}"
        log_line += f" | reward = {record.reward:.6f}"
        logger(log_line)
    final_front = front
    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "batch": int(getattr(args, "batch", 1)),
        "surrogate_model": surrogate_model_name(args),
        "candidate_solver": str(getattr(args, "solver", "moead_ego")).lower(),
        "reward_lambda": float(args.reward_lambda),
        "reward_scheme": int(reward_scheme_id),
        "agent_name": args.agent_name,
        "policy_name": policy_name.lower(),
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": hv_history[-1],
        "final_igd": igd_history[-1],
        "archive_x": archive_x.tolist(),
        "archive_f": archive_y.tolist(),
        "archive_g": archive_g.tolist(),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "history": [asdict(item) for item in history],
        "fe_history": fe_history,
        "hv_history": hv_history,
        "igd_history": igd_history,
    }
    if make_plot:
        curve = aggregate_metric_histories([summary])
        summary["plot_path"], _ = plot_metric_curves(
            args, [(primary_policy_display_name(policy_name), curve, summary)],
            true_pareto, f"seed{args.seed}",
        )
    return summary, archive_y


def build_initial_archive(
    *,
    args: argparse.Namespace,
    problem,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(args.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(seed),
    )
    archive_y, archive_g = evaluate_objectives_and_constraints(problem, archive_x)
    return np.asarray(archive_x, dtype=np.float32), archive_y, archive_g


def run_policy_rollout_once(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    policy_name: str,
    disc: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    logger=print,
    run_label: str | None = None,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
    initial_archive: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    del run_label
    run_started_at = time.perf_counter()
    if initial_archive is None:
        initial_archive = build_initial_archive(
            args=args, problem=problem, seed=int(args.seed),
        )
    archive_x_init, archive_y_init, archive_g_init = initial_archive
    summary, archive_y = run_policy_rollout(
        args=args,
        problem=problem,
        nsga2_problem=nsga2_problem,
        ref_point=ref_point,
        true_pareto=true_pareto,
        archive_x_init=archive_x_init,
        archive_y_init=archive_y_init,
        archive_g_init=archive_g_init,
        policy_name=policy_name,
        disc=disc,
        infill_criterion=infill_criterion,
        compare_mode=compare_mode,
        make_plot=False,
        logger=logger,
        reward_scheme_id=int(reward_scheme_id),
        true_pareto_hv=true_pareto_hv,
    )
    summary = dict(summary)
    summary["wall_clock_sec"] = float(time.perf_counter() - run_started_at)
    return summary, np.asarray(archive_y, dtype=np.float32)


def build_plot_command_caption() -> str:
    command_text = "python " + " ".join(str(part) for part in sys.argv[1:])
    wrapped = textwrap.wrap(command_text, width=120)
    return "\n".join(wrapped)


def attach_plot_caption(fig) -> None:
    caption = build_plot_command_caption()
    fig.text(
        0.01,
        0.01,
        caption,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )


def show_plot(fig) -> None:
    try:
        width_in, height_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "resize"):
            manager.resize(int(width_in * dpi), int(height_in * dpi))
    except Exception:
        pass
    plt.show()


def resolve_pdf_plot_path(plot_file: Path, args: argparse.Namespace) -> Path:
    return default_test_plot_dir(args, kind="pdf") / f"{plot_file.stem}.pdf"


def save_plot_outputs(
    fig,
    *,
    plot_file: Path,
    args: argparse.Namespace,
    with_command_caption_on_png: bool,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    pdf_file = resolve_pdf_plot_path(plot_file, args)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(pdf_file)
    if with_command_caption_on_png:
        attach_plot_caption(fig)
        # Do not reserve extra bottom space here: it changes the effective plot area
        # away from the requested 4:3 figure ratio.
    fig.savefig(plot_file, dpi=180)
    return pdf_file.resolve()



def aggregate_metric_histories(runs: list[dict]) -> dict:
    """Aggregate matched evaluations and keep the IGD sample count at each FE."""
    fe_history = runs[0]["fe_history"]
    if any(run["fe_history"] != fe_history for run in runs):
        raise ValueError("Runs must share the same FE history before aggregation.")
    hv = np.asarray([run["hv_history"] for run in runs], dtype=float)
    igd = np.asarray([run["igd_history"] for run in runs], dtype=float)
    igd_mean, igd_std, igd_count = [], [], []
    for column in igd.T:
        valid = column[np.isfinite(column)]
        igd_mean.append(float(valid.mean()) if len(valid) else None)
        igd_std.append(float(valid.std()) if len(valid) else None)
        igd_count.append(len(valid))
    return {
        "fe_history": fe_history,
        "hv_history": hv.mean(axis=0).tolist(),
        "hv_std_history": hv.std(axis=0).tolist(),
        "igd_history": igd_mean,
        "igd_std_history": igd_std,
        "igd_valid_count": igd_count,
        "final_hv": float(hv[:, -1].mean()),
        "final_igd": igd_mean[-1],
    }


def plot_metric_curves(args, series, true_pareto, run_tag):
    """Write HV and IGD curves and a Pareto view of the first seed."""
    agent_tag = resolve_primary_output_tag(args)
    plot_file = (
        Path(args.plot_path) if getattr(args, "plot_path", None)
        else default_test_plot_dir(args, kind="png") / f"test_{agent_tag}_{args.problem.lower()}_{run_tag}.png"
    )
    fig = plt.figure(figsize=(17, 5.2))
    hv_ax, igd_ax = fig.add_subplot(131), fig.add_subplot(132)
    n_obj = true_pareto.shape[1] if true_pareto is not None else len(series[0][2]["reference_point"])
    pareto_ax = fig.add_subplot(133, projection="3d" if n_obj == 3 else None)
    colors = plt.get_cmap("tab20")
    for i, (label, curve, first_run) in enumerate(series):
        color = colors(i % colors.N)
        for ax, metric in ((hv_ax, "hv"), (igd_ax, "igd")):
            average = np.asarray(curve[f"{metric}_history"], dtype=float)
            std = np.asarray(curve[f"{metric}_std_history"], dtype=float)
            fe = curve["fe_history"]
            ax.plot(fe, average, label=label, color=color)
            ax.fill_between(fe, average - std, average + std, color=color, alpha=0.18)
        front = np.asarray(first_run["final_front"], dtype=float).reshape(-1, n_obj)
        if len(front):
            pareto_ax.scatter(*front.T, color=color, s=15, label=label)
    for ax, title in ((hv_ax, "HV"), (igd_ax, "IGD")):
        ax.set(xlabel="FE", ylabel=title, title=f"{title} mean ? std")
        ax.grid(alpha=0.3)
    hv_ax.legend()
    if true_pareto is not None:
        pareto_ax.scatter(*true_pareto.T, color="grey", s=3, alpha=0.35, label="Reference front")
    pareto_ax.set(xlabel="F1", ylabel="F2", title=f"Feasible front, seed {series[0][2]['seed']}")
    if n_obj == 3:
        pareto_ax.set_zlabel("F3")
    if pareto_ax.get_legend_handles_labels()[0]:
        pareto_ax.legend()
    try:
        pdf_file = save_plot_outputs(fig, plot_file=plot_file, args=args, with_command_caption_on_png=False)
        show_plot(fig)
    finally:
        plt.close(fig)
    return str(plot_file.resolve()), str(pdf_file)


def main(agent_name: str = "nhis") -> None:
    args = parse_args()
    args.agent_name = str(agent_name).lower()
    primary = resolve_primary_policy_name(args)
    if is_agent_policy(primary):
        args.agent_name = primary
        if not args.random_model and not args.agent_pth:
            args.agent_pth = resolve_default_best_reward_checkpoint(args)
    seeds = args.seeds if args.seeds is not None else [resolve_seed(args.seed)]
    infill_names = resolve_compare_infill_names(args)
    algo_names = resolve_compare_algo_names(args)
    problem = make_problem(args.problem, dim=args.dim)
    true_pareto = reference_front(args.problem)
    adapter = make_nsga2_problem_adapter(problem, problem.n_obj)
    reward_scheme_id = resolve_test_reward_scheme(args)
    if args.reward_lambda is None:
        args.reward_lambda = resolve_default_reward_lambda(reward_scheme_id)

    set_seed(seeds[0])
    primary_agent = build_agent(args, map_location=args.device, agent_name=primary)[0] if is_agent_policy(primary) else None
    cases = [("primary", primary, primary_policy_display_name(primary), args, primary_agent)]
    for name in infill_names:
        cases.append(("infill", name, compare_infill_display_name(name), args, None))
    for name in algo_names:
        compare_args = argparse.Namespace(**vars(args))
        model = None
        if is_agent_policy(name):
            compare_args.agent_name = name
            compare_args.agent_pth = args.compare_agent_pth
            compare_args.random_model = args.compare_random_model or (args.random_model and not args.compare_agent_pth)
            if not compare_args.random_model and not compare_args.agent_pth:
                raise ValueError(f"compare_algo={name} requires --compare_agent_pth or a random model.")
            model, _ = build_agent(compare_args, map_location=args.device, agent_name=name)
        cases.append(("algo", name, compare_algo_display_name(name), compare_args, model))

    histories = {(group, name): [] for group, name, _, _, _ in cases}
    per_seed = []
    for seed in seeds:
        args.seed = seed
        set_seed(seed)
        log_path = default_test_log_path(args, agent_name=args.agent_name)
        log, log_fp = make_test_logger(log_path)
        try:
            initial_archive = build_initial_archive(args=args, problem=problem, seed=seed)
            ref_point = get_reference_point(initial_archive[1])
            true_hv = hypervolume(true_pareto, ref_point) if reward_scheme_id == 2 else None
            log(f"test | problem = {args.problem} | dim = {args.dim} | seed = {seed} | solver = {args.solver}")
            log(f"reference_point = {ref_point.tolist()}")
            seed_result = {
                "seed": seed, "reference_point": ref_point.tolist(), "test_log_path": str(log_path.resolve()),
                "infill_results": {}, "baseline_results": {},
            }
            for group, name, label, case_args, model in cases:
                case_args.seed = seed
                set_seed(seed)
                criterion = None if model is not None or name == "moead_ego" else build_compare_infill_criterion(
                    name, ref_point=ref_point
                )
                result, _ = run_policy_rollout_once(
                    args=case_args, problem=problem, nsga2_problem=adapter, ref_point=ref_point,
                    true_pareto=true_pareto, policy_name=name, disc=model, infill_criterion=criterion,
                    compare_mode=len(cases) > 1, logger=log, run_label=label, reward_scheme_id=reward_scheme_id,
                    true_pareto_hv=true_hv, initial_archive=initial_archive,
                )
                histories[group, name].append(result)
                if group == "primary":
                    seed_result["primary"] = result
                else:
                    key = "infill_results" if group == "infill" else "baseline_results"
                    seed_result[key][name] = result
                log(f"{label} | final HV = {result['final_hv']:.12f} | final IGD = {format_igd(result['final_igd'])}")
            per_seed.append(seed_result)
        finally:
            log_fp.close()

    aggregate = {key: aggregate_metric_histories(runs) for key, runs in histories.items()}
    run_tag = f"seed{seeds[0]}" if len(seeds) == 1 else f"seeds{len(seeds)}_{datetime.now():%Y%m%d_%H%M%S}"
    if infill_names or algo_names:
        run_tag += "_compare"
    series = [(label, aggregate[group, name], histories[group, name][0]) for group, name, label, _, _ in cases]
    plot_path, pdf_path = plot_metric_curves(args, series, true_pareto, run_tag)
    primary_metrics = aggregate["primary", primary]
    summary = {
        "problem": args.problem, "dim": args.dim, "seeds": seeds,
        "agent_name": args.agent_name, "primary_policy": primary, "reward_scheme": reward_scheme_id,
        "compare_infill_resolved": infill_names, "compare_algo_resolved": algo_names,
        "reference_front": true_pareto.tolist(),
        "primary": primary_metrics,
        "infill_results": {name: aggregate["infill", name] for name in infill_names},
        "baseline_results": {name: aggregate["algo", name] for name in algo_names},
        "runs": per_seed, "plot_path": plot_path, "pdf_plot_path": pdf_path,
        **primary_metrics,
    }
    if len(seeds) == 1:
        summary.update(seed=seeds[0], reference_point=per_seed[0]["reference_point"],
                       test_log_path=per_seed[0]["test_log_path"])
    output = (Path(args.output_json) if args.output_json
              else Path(per_seed[0]["test_log_path"]).with_suffix(".json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
