from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from lhs import latin_hypercube_sample
from solver.population_utils import sample_unique_non_parent, unique_non_parent_indices
from solver.solver import _ModelListSurrogate, _surrogate_predict_mean


@dataclass(frozen=True)
class NSGA2Result:
    x: np.ndarray
    mean: np.ndarray
    constraints: np.ndarray


class _SurrogateProblem(Problem):
    def __init__(self, surrogate, constraint_surrogate, **problem_kwargs):
        super().__init__(**problem_kwargs)
        self.surrogate = surrogate
        self.constraint_surrogate = constraint_surrogate

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._predict(self.surrogate, x, self.n_obj, "Objective")
        if self.n_ieq_constr:
            out["G"] = self._predict(self.constraint_surrogate, x, self.n_ieq_constr, "Constraint")

    @staticmethod
    def _predict(surrogate, x, width, label):
        values = _surrogate_predict_mean(surrogate, x)
        if values.shape != (len(x), width) or not np.isfinite(values).all():
            raise ValueError(f"{label} surrogate must return a finite matrix of shape {(len(x), width)}.")
        return values


def run_surrogate_nsga2(
    problem,
    archive_x,
    pop_size,
    gps=None,
    surrogate=None,
    saea_steps=25,
    seed=0,
    *,
    constraint_surrogate=None,
    seed_x=None,
) -> NSGA2Result:
    """Return new candidates ranked using predicted objectives and G <= 0.

    NSGA-II initializes from sampled archive points or explicit seed_x and
    fills remaining slots with LHS. All evaluations use the supplied surrogates.
    """
    pop_size, saea_steps = int(pop_size), int(saea_steps)
    if pop_size < 2 or saea_steps < 1:
        raise ValueError("NSGA-II requires pop_size >= 2 and saea_steps >= 1.")
    if surrogate is None:
        if gps is None:
            raise ValueError("Supply an objective surrogate or gps for NSGA-II.")
        surrogate = _ModelListSurrogate(gps)

    n_var = getattr(problem, "n_var", getattr(problem, "dim", None))
    lower = np.broadcast_to(getattr(problem, "xl", getattr(problem, "lower", None)), (n_var,)).astype(float)
    upper = np.broadcast_to(getattr(problem, "xu", getattr(problem, "upper", None)), (n_var,)).astype(float)
    archive = np.asarray(archive_x, dtype=np.float64)
    if archive.ndim != 2 or archive.shape[1] != n_var:
        raise ValueError(f"archive_x must have shape [N, {n_var}].")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("NSGA-II requires finite, ordered bounds.")
    rng = np.random.default_rng(seed)
    if seed_x is None:
        seeds = archive[unique_non_parent_indices(archive, archive[:0])]
        initial = seeds[rng.choice(len(seeds), size=min(len(seeds), pop_size), replace=False)]
    else:
        seeds = np.asarray(seed_x, dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape[1] != n_var or len(seeds) > pop_size:
            raise ValueError(f"seed_x must have shape [K, {n_var}] with K <= pop_size.")
        initial = seeds[unique_non_parent_indices(seeds, archive[:0])]
    if not np.isfinite(initial).all() or np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial NSGA-II seeds must be finite and within bounds.")
    if len(initial) < pop_size:
        unit = latin_hypercube_sample(
            n_samples=pop_size - len(initial), dim=n_var, lower=0.0, upper=1.0,
            seed=int(rng.integers(2**32)),
        )
        initial = np.vstack((initial, lower + unit * (upper - lower)))

    definition = getattr(problem, "pymoo_problem", problem)
    if getattr(definition, "n_eq_constr", 0):
        raise ValueError("NSGA-II accepts inequality constraints G <= 0.")
    n_constraints = getattr(definition, "n_ieq_constr", None)
    if constraint_surrogate is None:
        if n_constraints:
            raise ValueError("Supply constraint_surrogate for a constrained problem.")
        n_constraints = 0
    elif n_constraints is None:
        n_constraints = _surrogate_predict_mean(constraint_surrogate, initial[:1]).shape[1]
    if not n_constraints and constraint_surrogate is not None:
        raise ValueError("The problem declares no inequality constraints for the constraint surrogate.")
    proxy = _SurrogateProblem(
        surrogate, constraint_surrogate, n_var=n_var, n_obj=int(problem.n_obj),
        n_ieq_constr=n_constraints, xl=lower, xu=upper,
    )
    algorithm = NSGA2(pop_size=pop_size, sampling=initial, eliminate_duplicates=True)
    result = minimize(proxy, algorithm, ("n_gen", saea_steps), seed=int(seed), verbose=False)

    population = result.pop
    candidates = population[unique_non_parent_indices(population.get("X"), archive)]
    if len(candidates) < pop_size:
        existing = np.asarray(candidates.get("X")).reshape(-1, n_var)
        fill = sample_unique_non_parent(
            lower=lower, upper=upper, archive_x=archive, existing_x=existing,
            n_samples=pop_size - len(candidates), rng=rng,
        )
        if len(unique_non_parent_indices(fill, np.vstack((archive, existing)))) != len(fill):
            raise ValueError("Could not sample enough distinct candidates outside the archive.")
        additions = Population.new("X", fill)
        result.algorithm.evaluator.eval(proxy, additions)
        candidates = Population.merge(candidates, additions)
    selected = result.algorithm.survival.do(proxy, candidates, n_survive=pop_size, random_state=rng)
    return NSGA2Result(
        x=np.asarray(selected.get("X")),
        mean=np.asarray(selected.get("F")),
        constraints=np.asarray(selected.get("G")).reshape(len(selected), n_constraints),
    )
