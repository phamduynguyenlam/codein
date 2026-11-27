from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from reward import pareto_front, pareto_improvement_scores, resolve_default_reward_lambda


def _import_botorch_ehvi():
    try:
        import torch
        from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
        from botorch.models.model import Model
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
        from gpytorch.distributions import MultitaskMultivariateNormal
    except Exception as exc:
        raise ImportError(
            "EHVI requires botorch and gpytorch. Install the dependencies from requirements.txt."
        ) from exc

    return {
        "torch": torch,
        "Model": Model,
        "GPyTorchPosterior": GPyTorchPosterior,
        "MultitaskMultivariateNormal": MultitaskMultivariateNormal,
        "NondominatedPartitioning": NondominatedPartitioning,
        "qExpectedHypervolumeImprovement": qExpectedHypervolumeImprovement,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
    }


def _build_botorch_candidate_model(candidate_mean, candidate_std, botorch_mod):
    torch = botorch_mod["torch"]
    model_base = botorch_mod["Model"]
    posterior_cls = botorch_mod["GPyTorchPosterior"]
    multitask_mvn_cls = botorch_mod["MultitaskMultivariateNormal"]

    class CandidateNormalModel(model_base):
        """Expose precomputed independent Gaussian predictions as a BoTorch model."""

        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("candidate_mean", mean)
            self.register_buffer("candidate_variance", std.square())

        @property
        def num_outputs(self):
            return int(self.candidate_mean.shape[-1])

        @property
        def batch_shape(self):
            return torch.Size()

        def posterior(
            self,
            X,
            output_indices=None,
            observation_noise=False,
            posterior_transform=None,
            **kwargs,
        ):
            del observation_noise, kwargs
            indices = X[..., 0].round().long()
            mean = self.candidate_mean[indices]
            variance = self.candidate_variance[indices]
            if output_indices is not None:
                mean = mean[..., output_indices]
                variance = variance[..., output_indices]
            covariance = torch.diag_embed(variance.flatten(start_dim=-2))
            posterior = posterior_cls(multitask_mvn_cls(mean, covariance))
            if posterior_transform is not None:
                posterior = posterior_transform(posterior)
            return posterior

    return CandidateNormalModel(candidate_mean, candidate_std)


class InfillCriterion(ABC):
    @abstractmethod
    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray: ...

    def select_index(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> tuple[int, np.ndarray]:
        scores = np.asarray(
            self.score_candidates(
                archive_y=np.asarray(archive_y, dtype=np.float32),
                candidate_mean=np.asarray(candidate_mean, dtype=np.float32),
                candidate_std=np.asarray(candidate_std, dtype=np.float32),
                seed=seed,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if scores.size <= 0:
            raise ValueError("InfillCriterion received no candidate scores.")
        return int(np.argmax(scores)), scores


class RandomSelection(InfillCriterion):
    """Select uniformly from the current surrogate candidate set."""

    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        del archive_y, candidate_std
        n_candidates = int(np.asarray(candidate_mean).shape[0])
        if n_candidates <= 0:
            raise ValueError("RandomSelection received no candidates.")
        rng = np.random.default_rng(seed)
        # The argmax of i.i.d. continuous uniform samples is uniform over indices.
        return rng.random(n_candidates, dtype=np.float32)


class SurrogateParetoImprovement(InfillCriterion):
    """Score each surrogate mean using the Pareto-improvement proxy."""

    def __init__(self, *, reward_lambda: float | None = None):
        self.reward_lambda = float(
            resolve_default_reward_lambda() if reward_lambda is None else reward_lambda
        )

    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        del candidate_std, seed
        archive = np.asarray(archive_y, dtype=np.float32)
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        if archive.ndim != 2:
            raise ValueError(f"archive_y must be 2D, got shape={archive.shape}.")
        if cand_mean.ndim != 2:
            raise ValueError(f"candidate_mean must be 2D, got shape={cand_mean.shape}.")
        if int(cand_mean.shape[0]) == 0:
            return np.empty(0, dtype=np.float32)
        if int(archive.shape[0]) > 0 and int(archive.shape[1]) != int(cand_mean.shape[1]):
            raise ValueError(
                f"archive_y and candidate_mean must have the same objective count, "
                f"got {archive.shape[1]} and {cand_mean.shape[1]}."
            )

        return pareto_improvement_scores(pareto_front(archive), cand_mean, self.reward_lambda)


class ExpectedHypervolumeImprovement(InfillCriterion):
    def __init__(
        self,
        *,
        ref_point: np.ndarray,
        n_samples: int = 128,
        min_std: float = 1e-6,
    ):
        self.ref_point = np.asarray(ref_point, dtype=np.float32).reshape(-1)
        self.n_samples = max(1, int(n_samples))
        self.min_std = float(max(min_std, 1e-12))

    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = np.asarray(candidate_std, dtype=np.float32)
        if cand_mean.ndim != 2:
            raise ValueError(f"candidate_mean must be 2D, got shape={cand_mean.shape}.")
        if cand_std.ndim == 1:
            cand_std = cand_std.reshape(-1, 1)
        if cand_std.shape != cand_mean.shape:
            raise ValueError(
                f"candidate_std must match candidate_mean shape, got {cand_std.shape} vs {cand_mean.shape}."
            )

        if int(cand_mean.shape[0]) == 0:
            return np.empty(0, dtype=np.float32)
        if int(cand_mean.shape[1]) != int(self.ref_point.shape[0]):
            raise ValueError(
                f"ref_point has {self.ref_point.shape[0]} objectives, "
                f"but candidates have {cand_mean.shape[1]}."
            )

        botorch_mod = _import_botorch_ehvi()
        torch = botorch_mod["torch"]
        dtype = torch.double
        device = torch.device("cpu")

        # BoTorch maximizes objectives; the repository's problems are minimization tasks.
        archive_t = -torch.as_tensor(archive_front, dtype=dtype, device=device)
        mean_t = -torch.as_tensor(cand_mean, dtype=dtype, device=device)
        std_t = torch.as_tensor(
            np.maximum(cand_std, self.min_std), dtype=dtype, device=device
        )
        ref_point_t = -torch.as_tensor(self.ref_point, dtype=dtype, device=device)

        model = _build_botorch_candidate_model(mean_t, std_t, botorch_mod)
        partitioning = botorch_mod["NondominatedPartitioning"](
            ref_point=ref_point_t,
            Y=archive_t,
        )
        sampler = botorch_mod["SobolQMCNormalSampler"](
            sample_shape=torch.Size([self.n_samples]),
            seed=None if seed is None else int(seed),
        )
        acquisition = botorch_mod["qExpectedHypervolumeImprovement"](
            model=model,
            ref_point=ref_point_t.tolist(),
            partitioning=partitioning,
            sampler=sampler,
        )
        candidate_indices = torch.arange(
            int(cand_mean.shape[0]), dtype=dtype, device=device
        ).reshape(-1, 1, 1)
        with torch.no_grad():
            scores = acquisition(candidate_indices)
        return scores.detach().cpu().numpy().astype(np.float32).reshape(-1)


class USeMOUncertainty(InfillCriterion):
    """Restrict selection to surrogate PF candidates, then maximize uncertainty."""

    def score_candidates(
        self,
        *,
        archive_y: np.ndarray,
        candidate_mean: np.ndarray,
        candidate_std: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        del archive_y, seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        uncertainty = np.linalg.norm(cand_std, axis=1).astype(np.float32)

        surrogate_front = np.asarray(pareto_front(cand_mean), dtype=np.float32)
        nd_mask = np.zeros(int(cand_mean.shape[0]), dtype=bool)
        for idx, candidate in enumerate(cand_mean):
            matched = np.all(np.isclose(surrogate_front, candidate[None, :], rtol=0.0, atol=1e-7), axis=1)
            nd_mask[idx] = bool(np.any(matched))

        scores = np.full(int(cand_mean.shape[0]), -np.inf, dtype=np.float32)
        scores[nd_mask] = uncertainty[nd_mask]
        return scores.astype(np.float32)


def _normalize_scalar(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    span = max(max_v - min_v, 1e-12)
    return ((arr - min_v) / span).astype(np.float32)


def _normalize_objectives(values: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    span = np.maximum(maxs_arr - mins_arr, 1e-12)
    return (values_arr - mins_arr) / span


def _vector_angle(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    denom = float(np.linalg.norm(x_arr) * np.linalg.norm(y_arr))
    if denom <= 1e-12:
        return 0.0
    cos_val = float(np.clip(float(np.dot(x_arr, y_arr)) / denom, -1.0, 1.0))
    return float(np.arccos(cos_val))


def _simplex_reference_vectors(n_obj: int, n_partitions: int) -> np.ndarray:
    if int(n_obj) <= 1:
        return np.ones((1, 1), dtype=np.float32)
    ref_vectors = np.asarray(
        get_reference_directions(
            "das-dennis", int(n_obj), n_partitions=max(1, int(n_partitions))
        ),
        dtype=np.float32,
    )
    norms = np.linalg.norm(ref_vectors, axis=1, keepdims=True)
    return ref_vectors / np.maximum(norms, 1e-12)


def _normalize_for_pbi(values: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    all_values = np.vstack([reference_values, values]).astype(np.float32)
    mins = all_values.min(axis=0)
    spans = np.maximum(all_values.max(axis=0) - mins, 1e-12)
    return (np.asarray(values, dtype=np.float32) - mins) / spans


def _pbi_stats(normalized_values: np.ndarray, ref_vectors: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1_all = normalized_values @ ref_vectors.T
    proj = d1_all[..., None] * ref_vectors[None, :, :]
    diff = normalized_values[:, None, :] - proj
    d2_all = np.linalg.norm(diff, axis=2)
    assoc = np.argmin(d2_all, axis=1)
    row_idx = np.arange(normalized_values.shape[0], dtype=np.int64)
    d1 = d1_all[row_idx, assoc]
    d2 = d2_all[row_idx, assoc]
    pbi = d1 + float(theta) * d2
    return assoc.astype(np.int64), d1.astype(np.float32), pbi.astype(np.float32)


def _random_unit_reference_vector(n_obj: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.random(int(n_obj), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return np.full(int(n_obj), 1.0 / np.sqrt(max(int(n_obj), 1)), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _pd_value(values: np.ndarray, ref_vector: np.ndarray, theta: float = 5.0) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    ref_vector_arr = np.asarray(ref_vector, dtype=np.float32)
    ref_norm = np.linalg.norm(ref_vector_arr)
    if ref_norm <= 1e-12:
        ref_vector_arr = np.full(ref_vector_arr.shape[0], 1.0 / np.sqrt(max(ref_vector_arr.shape[0], 1)), dtype=np.float32)
        ref_norm = np.linalg.norm(ref_vector_arr)

    d1 = (values_arr @ ref_vector_arr) / max(ref_norm, 1e-12)
    projection = (d1 / max(ref_norm, 1e-12))[:, None] * ref_vector_arr[None, :]
    d2 = np.linalg.norm(values_arr - projection, axis=1)
    return d1 + float(theta) * d2


def _nd_a_components(candidate_values: np.ndarray, archive_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_values_arr = np.asarray(candidate_values, dtype=np.float32)
    archive_front = pareto_front(np.asarray(archive_values, dtype=np.float32))
    combined = np.vstack([candidate_values_arr, archive_front]).astype(np.float32)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    candidate_norm = _normalize_objectives(candidate_values_arr, mins, maxs)
    archive_norm = _normalize_objectives(archive_front, mins, maxs)

    angles = np.zeros(candidate_values_arr.shape[0], dtype=np.float32)
    distances = np.zeros(candidate_values_arr.shape[0], dtype=np.float32)
    for idx, candidate in enumerate(candidate_norm):
        if archive_norm.shape[0] == 0:
            angles[idx] = 0.0
            distances[idx] = 0.0
        else:
            angle_values = np.asarray([_vector_angle(candidate, archive_point) for archive_point in archive_norm], dtype=np.float32)
            distance_values = np.linalg.norm(archive_norm - candidate[None, :], axis=1).astype(np.float32)
            angles[idx] = float(angle_values.min())
            distances[idx] = float(distance_values.min())
    return angles, distances


def _nd_pbi_branch_params(focus: str) -> tuple[float, float]:
    focus_name = str(focus).lower()
    if focus_name == "convergence":
        return 2.0, 0.0
    if focus_name == "diversity":
        return 8.0, 0.5
    raise ValueError(f"Unsupported ND-PBI focus: {focus}")


def _ensure_sigma_shape(candidate_mean: np.ndarray, candidate_std: np.ndarray | None) -> np.ndarray:
    cand_mean = np.asarray(candidate_mean, dtype=np.float32)
    if candidate_std is None:
        return np.full_like(cand_mean, 1e-6, dtype=np.float32)
    cand_std_arr = np.asarray(candidate_std, dtype=np.float32)
    if cand_std_arr.ndim == 1:
        cand_std_arr = cand_std_arr.reshape(-1, 1)
    if cand_std_arr.shape[1] == 1 and cand_mean.shape[1] > 1:
        cand_std_arr = np.repeat(cand_std_arr, cand_mean.shape[1], axis=1)
    if cand_std_arr.shape != cand_mean.shape:
        raise ValueError(f"candidate_std must match candidate_mean shape, got {cand_std_arr.shape} vs {cand_mean.shape}.")
    return cand_std_arr.astype(np.float32)


class NDA(InfillCriterion):
    def __init__(self, *, diversity_lambda: float = 1.0):
        self.diversity_lambda = float(diversity_lambda)

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        penalized_values = cand_mean + cand_std
        angles, _ = _nd_a_components(penalized_values, archive_y)
        uncertainty = _normalize_scalar(cand_std.mean(axis=1))
        return (angles + self.diversity_lambda * uncertainty).astype(np.float32)


class NDPBIConvergence(InfillCriterion):
    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        theta, empty_bonus = _nd_pbi_branch_params("convergence")
        penalized = cand_mean + cand_std
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        reference_values = np.vstack([archive_front, penalized]).astype(np.float32)
        ref_vectors = _simplex_reference_vectors(penalized.shape[1], n_partitions=max(12, penalized.shape[1] * 4))

        arnd_norm = _normalize_for_pbi(archive_front, reference_values)
        cand_norm = _normalize_for_pbi(penalized, reference_values)
        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, cand_d1, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
        scores = np.zeros(penalized.shape[0], dtype=np.float32)
        for idx in range(penalized.shape[0]):
            assoc = int(cand_assoc[idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[idx])
            else:
                improvement = float(empty_bonus) - float(cand_pbi[idx])
            scores[idx] = float(improvement - 0.05 * float(cand_d1[idx]))
        return scores.astype(np.float32)


class NDPBIDiversity(InfillCriterion):
    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        theta, empty_bonus = _nd_pbi_branch_params("diversity")
        penalized = cand_mean + cand_std
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        reference_values = np.vstack([archive_front, penalized]).astype(np.float32)
        ref_vectors = _simplex_reference_vectors(penalized.shape[1], n_partitions=max(12, penalized.shape[1] * 4))

        arnd_norm = _normalize_for_pbi(archive_front, reference_values)
        cand_norm = _normalize_for_pbi(penalized, reference_values)
        arnd_assoc, _, arnd_pbi = _pbi_stats(arnd_norm, ref_vectors, theta=theta)
        cand_assoc, _, cand_pbi = _pbi_stats(cand_norm, ref_vectors, theta=theta)

        nonempty_refs = set(int(idx) for idx in arnd_assoc.tolist())
        scores = np.zeros(penalized.shape[0], dtype=np.float32)
        for idx in range(penalized.shape[0]):
            assoc = int(cand_assoc[idx])
            if assoc in nonempty_refs:
                ref_mask = arnd_assoc == assoc
                pbi_min = float(np.min(arnd_pbi[ref_mask]))
                improvement = pbi_min - float(cand_pbi[idx])
            else:
                improvement = float(empty_bonus) - float(cand_pbi[idx])
            scores[idx] = float(improvement + float(empty_bonus))
        return scores.astype(np.float32)


class EPDIExploitation(InfillCriterion):
    def __init__(self, *, mc_samples: int = 1000):
        self.mc_samples = max(1, int(mc_samples))

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        combined = np.vstack([archive_front, cand_mean]).astype(np.float32)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = _normalize_objectives(archive_front, mins, maxs)
        candidate_norm = _normalize_objectives(cand_mean, mins, maxs)
        sigma_norm = cand_std / np.maximum(maxs - mins, 1e-12)

        rng = np.random.default_rng(seed)
        mean_epdi = np.zeros(candidate_norm.shape[0], dtype=np.float32)
        for idx in range(candidate_norm.shape[0]):
            ref_vector = _random_unit_reference_vector(candidate_norm.shape[1], rng)
            pd_min = float(np.min(_pd_value(archive_norm, ref_vector))) if archive_norm.size else 0.0
            sigma = np.maximum(sigma_norm[idx], 1e-6)
            samples = rng.normal(loc=candidate_norm[idx], scale=sigma, size=(self.mc_samples, candidate_norm.shape[1])).astype(np.float32)
            samples = np.clip(samples, 0.0, 1.5)
            pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
            mean_epdi[idx] = float(np.mean(pdi_samples))
        return mean_epdi.astype(np.float32)


class EPDIExploration(InfillCriterion):
    def __init__(self, *, mc_samples: int = 1000):
        self.mc_samples = max(1, int(mc_samples))

    def score_candidates(self, *, archive_y: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray, seed: int | None = None) -> np.ndarray:
        cand_mean = np.asarray(candidate_mean, dtype=np.float32)
        cand_std = _ensure_sigma_shape(cand_mean, candidate_std)
        archive_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        combined = np.vstack([archive_front, cand_mean]).astype(np.float32)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        archive_norm = _normalize_objectives(archive_front, mins, maxs)
        candidate_norm = _normalize_objectives(cand_mean, mins, maxs)
        sigma_norm = cand_std / np.maximum(maxs - mins, 1e-12)

        rng = np.random.default_rng(seed)
        scores = np.zeros(candidate_norm.shape[0], dtype=np.float32)
        for idx in range(candidate_norm.shape[0]):
            ref_vector = _random_unit_reference_vector(candidate_norm.shape[1], rng)
            pd_min = float(np.min(_pd_value(archive_norm, ref_vector))) if archive_norm.size else 0.0
            sigma = np.maximum(sigma_norm[idx], 1e-6)
            samples = rng.normal(loc=candidate_norm[idx], scale=sigma, size=(self.mc_samples, candidate_norm.shape[1])).astype(np.float32)
            samples = np.clip(samples, 0.0, 1.5)
            pdi_samples = np.maximum(pd_min - _pd_value(samples, ref_vector), 0.0)
            scores[idx] = float(np.mean(pdi_samples) + np.std(pdi_samples))
        return scores.astype(np.float32)
