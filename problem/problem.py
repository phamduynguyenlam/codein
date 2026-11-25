
from __future__ import annotations

from functools import lru_cache
import re

import numpy as np
from pymoo.problems import get_problem

MW_PROBLEMS = tuple(f"MW{i}" for i in range(1, 15))
DAS_PROBLEMS = tuple(f"DAS-CMOP{i}" for i in range(1, 10))
SUPPORTED_PROBLEMS = [*MW_PROBLEMS, *DAS_PROBLEMS]
DEFAULT_TRAIN_DIM = 8
DEFAULT_TEST_DIM = 10


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
    problem_family(name)
    return DEFAULT_TEST_DIM


class BenchmarkProblem:
    def __init__(self, name: str, dim: int | None = None):
        self.name = normalize_problem_name(name)
        family = problem_family(self.name)
        self.dim = default_problem_dim(self.name) if dim is None else int(dim)
        if self.dim < 3:
            raise ValueError("MW and DAS-CMOP require at least 3 decision variables.")
        key = self.name.lower().replace("-", "")
        if family == "MW":
            self.pymoo_problem = get_problem(key, n_var=self.dim)
        else:
            # pymoo's DAS constructors fix n_var at 30. Their formulas use
            # n_var and the decision matrix, so also resize the bound arrays.
            self.pymoo_problem = get_problem(key, 1)
            self.pymoo_problem.n_var = self.dim
            self.pymoo_problem.xl = np.full(self.dim, self.pymoo_problem.xl[0])
            self.pymoo_problem.xu = np.full(self.dim, self.pymoo_problem.xu[0])
        self.n_obj = self.pymoo_problem.n_obj
        self.lower = float(self.pymoo_problem.xl[0])
        self.upper = float(self.pymoo_problem.xu[0])

    def evaluate(self, x):
        return self.pymoo_problem.evaluate(np.atleast_2d(x), return_values_of=["F"])

    def evaluate_with_constraints(self, x):
        return self.pymoo_problem.evaluate(np.atleast_2d(x), return_values_of=["F", "G"])

    def pareto_front(self):
        return reference_front(self.name, self.dim)


def make_problem(name: str, dim: int | None = None) -> BenchmarkProblem:
    return BenchmarkProblem(name, dim)


@lru_cache(maxsize=2 * len(SUPPORTED_PROBLEMS))
def _reference_front(name: str, dim: int) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    problem = make_problem(name, dim)
    try:
        with np.errstate(divide="ignore"):
            front = problem.pymoo_problem.pareto_front()
    except Exception as exc:
        raise RuntimeError(f"Could not load the pymoo reference front for {name}.") from exc

    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or not len(front) or front.shape[1] != problem.n_obj or not np.isfinite(front).all():
        raise ValueError(f"Invalid reference front for {name}.")
    front = np.unique(front, axis=0)
    return front[NonDominatedSorting().do(front, only_non_dominated_front=True)]


def reference_front(name: str, dim: int | None = None) -> np.ndarray:
    return _reference_front(normalize_problem_name(name), default_problem_dim(name) if dim is None else int(dim)).copy()


def get_reference_point(initial_objectives: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial_objectives, dtype=np.float64)
    if initial.ndim != 2 or not initial.size or not np.isfinite(initial).all():
        raise ValueError("Initial objective observations must be a finite nonempty matrix.")
    return 1.1 * initial.max(axis=0)
