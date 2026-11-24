from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


DEFAULT_REWARD = "pareto"
DEFAULT_REWARD_LAMBDAS = {"pareto": 5.0, "hv": 1.0}
REWARD_CHOICES = tuple(DEFAULT_REWARD_LAMBDAS)


def normalize_reward(reward: str) -> str:
    name = str(reward).strip().lower()
    name = {"1": "pareto", "2": "hv"}.get(name, name)
    if name not in DEFAULT_REWARD_LAMBDAS:
        raise ValueError(f"Unsupported reward: {reward}. Expected one of {REWARD_CHOICES}.")
    return name


def resolve_default_reward_lambda(reward: str = DEFAULT_REWARD) -> float:
    return DEFAULT_REWARD_LAMBDAS[normalize_reward(reward)]


def pareto_front(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values.shape}")
    if len(values) == 0:
        return values

    keep = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return values[np.asarray(keep, dtype=np.int64)]


def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
    front = pareto_front(values)
    if front.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))


def feasible_mask(constraints: np.ndarray) -> np.ndarray:
    return (constraints <= 0).all(axis=1)


def pareto_improvement_scores(previous_front, candidates, reward_lambda):
    """Score candidates independently using Euclidean distances to a fixed front."""
    front = np.asarray(previous_front, dtype=np.float64)
    candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    if not len(front):
        return np.ones(len(candidates), dtype=np.float32)
    scores = np.full(len(candidates), -1.0, dtype=np.float32)
    for i, candidate in enumerate(candidates):
        if ((front <= candidate).all(axis=1) & (front < candidate).any(axis=1)).any():
            continue
        distances = np.linalg.norm(front - candidate, axis=1)
        nearest = int(np.argmin(distances))
        d_ref = max(float(np.linalg.norm(front[nearest])), 1e-12)
        scores[i] = max(1e-6, 1.0 + float(reward_lambda) * distances[nearest] / d_ref)
    return scores


def _hv_improvement(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    true_pareto_hv: float,
    hv_epsilon: float = 1e-8,
    reward_lambda: float = DEFAULT_REWARD_LAMBDAS["hv"],
) -> float:
    """Normalized HV improvement against the remaining gap to the true Pareto-front HV."""
    previous_front = np.asarray(previous_front, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = np.vstack([previous_front, selected_objectives])

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(combined_front, ref_point)
    if next_hv <= prev_hv:
        return 0.0
    remaining_gap = max(float(true_pareto_hv) - float(next_hv), float(hv_epsilon))
    return float(float(reward_lambda) * (float(next_hv) - float(prev_hv)) / remaining_gap)


def compute_env_reward(
    *,
    previous_archive_y: np.ndarray,
    previous_archive_g: np.ndarray,
    selected_y: np.ndarray,
    selected_g: np.ndarray,
    constraint_scale: np.ndarray,
    ref_point: np.ndarray,
    reward: str = DEFAULT_REWARD,
    reward_lambda: float | None = None,
    true_pareto_hv: float | None = None,
) -> float:
    """Evaluate one selected point using observations from before its insertion."""
    reward = normalize_reward(reward)
    previous_feasible = feasible_mask(previous_archive_g)
    selected_g = np.atleast_2d(selected_g)
    (selected_is_feasible,) = feasible_mask(selected_g)
    if not previous_feasible.any():
        if selected_is_feasible:
            return 1.0
        if not len(previous_archive_g):
            return 0.0
        epsilon = 1e-8
        scale = np.maximum(np.asarray(constraint_scale, dtype=np.float64), epsilon)
        old_best = np.maximum(previous_archive_g / scale, 0.0).sum(axis=1).min()
        new_violation = np.maximum(selected_g / scale, 0.0).sum()
        new_best = min(old_best, new_violation)
        return float((old_best - new_best) / max(old_best, epsilon))

    if not selected_is_feasible:
        return -1.0

    front = pareto_front(previous_archive_y[previous_feasible])
    weight = resolve_default_reward_lambda(reward) if reward_lambda is None else float(reward_lambda)
    if reward == "pareto":
        return float(pareto_improvement_scores(front, selected_y, weight)[0])
    if true_pareto_hv is None:
        raise ValueError("The hv reward requires true_pareto_hv.")
    return _hv_improvement(
        previous_front=front, selected_objectives=np.atleast_2d(selected_y),
        ref_point=ref_point, true_pareto_hv=true_pareto_hv, reward_lambda=weight,
    )
