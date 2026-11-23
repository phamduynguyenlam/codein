
from __future__ import annotations

from functools import lru_cache
import re

import numpy as np
from pymoo.problems import get_problem

MW_PROBLEMS = tuple(f"MW{i}" for i in range(1, 15))
DAS_PROBLEMS = tuple(f"DAS-CMOP{i}" for i in range(1, 10))
SUPPORTED_PROBLEMS = [*MW_PROBLEMS, *DAS_PROBLEMS]


def normalize_problem_name(name: str) -> str:
    key = str(name).strip().upper()
    return re.sub(r"^DAS[-_]?CMOP", "DAS-CMOP", key)


def problem_family(name: str) -> str:
    key = normalize_problem_name(name)
    if key in MW_PROBLEMS:
        return "MW"
    if key in DAS_PROBLEMS:
        return "DAS"
    raise ValueError(f"Unsupported benchmark: {name}")


def default_problem_dim(name: str) -> int:
    return 15 if problem_family(name) == "MW" else 10


class BenchmarkProblem:
    def __init__(self, name: str, dim: int | None = None):
        self.name = normalize_problem_name(name)
        family = problem_family(self.name)
        self.dim = default_problem_dim(self.name)
        if dim is not None and int(dim) != self.dim:
            raise ValueError(f"{self.name} uses {self.dim} decision variables, got {dim}.")
        key = self.name.lower().replace("-", "")
        if family == "MW":
            self.pymoo_problem = get_problem(key)
        else:
            # pymoo's DAS constructors fix n_var at 30. Their formulas use
            # n_var and the decision matrix, so also resize the bound arrays.
            self.pymoo_problem = get_problem(key, 1)
            self.pymoo_problem.n_var = self.dim
            self.pymoo_problem.xl = self.pymoo_problem.xl[:self.dim].copy()
            self.pymoo_problem.xu = self.pymoo_problem.xu[:self.dim].copy()
        self.n_obj = self.pymoo_problem.n_obj
        self.lower = float(self.pymoo_problem.xl[0])
        self.upper = float(self.pymoo_problem.xu[0])

    def evaluate(self, x):
        return self.pymoo_problem.evaluate(np.atleast_2d(x), return_values_of=["F"])

    def evaluate_with_constraints(self, x):
        return self.pymoo_problem.evaluate(np.atleast_2d(x), return_values_of=["F", "G"])

    def pareto_front(self):
        return reference_front(self.name)


def make_problem(name: str, dim: int | None = None) -> BenchmarkProblem:
    return BenchmarkProblem(name, dim)


@lru_cache(maxsize=len(SUPPORTED_PROBLEMS))
def _reference_front(name: str) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    problem = make_problem(name)
    if problem_family(name) == "MW":
        try:
            with np.errstate(divide="ignore"):
                front = problem.pymoo_problem.pareto_front()
        except Exception as exc:
            raise RuntimeError(f"Could not load the pymoo reference front for {name}.") from exc
    else:
        lead_count = problem.n_obj - 1
        axis = np.linspace(0.0, 1.0, 4097 if lead_count == 1 else 257)
        lead = axis[:, None] if lead_count == 1 else np.column_stack(
            [grid.ravel() for grid in np.meshgrid(axis, axis, indexing="ij")]
        )
        x = np.empty((len(lead), problem.dim))
        x[:, :lead_count] = lead
        index = DAS_PROBLEMS.index(name) + 1
        # At difficulty 1 the tail minimizing g is feasible whenever the
        # angular constraints hold. Increasing g worsens every objective.
        if index <= 3:
            x[:, lead_count:] = np.sin(0.5 * np.pi * x[:, :1])
        elif index <= 8:
            x[:, lead_count:] = 0.5
        else:
            j = np.arange(lead_count + 1, problem.dim + 1)
            x[:, lead_count:] = np.cos(0.25 * np.pi * j / problem.dim * x[:, :2].sum(axis=1, keepdims=True))
        f, g = problem.evaluate_with_constraints(x)
        front = f[(g <= 1e-10).all(axis=1)]

    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or not len(front) or front.shape[1] != problem.n_obj or not np.isfinite(front).all():
        raise ValueError(f"Invalid reference front for {name}.")
    front = np.unique(front, axis=0)
    return front[NonDominatedSorting().do(front, only_non_dominated_front=True)]


def reference_front(name: str) -> np.ndarray:
    return _reference_front(normalize_problem_name(name)).copy()


def get_reference_point(initial_objectives: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial_objectives, dtype=np.float64)
    if initial.ndim != 2 or not initial.size or not np.isfinite(initial).all():
        raise ValueError("Initial objective observations must be a finite nonempty matrix.")
    return 1.1 * initial.max(axis=0)
