from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infill import _normalize_objectives, _pbi_stats, _vector_angle
from lhs import latin_hypercube_sample
from problem.constraints import evaluate_objectives_and_constraints
from problem.problem import SUPPORTED_PROBLEMS, make_problem, normalize_problem_name, reference_front, get_reference_point
from reward import feasible_mask, hypervolume, pareto_front
from solver.nsga3 import _build_reference_directions
from solver.solver import solve
from surrogate.surrogate import fit_gp_surrogates


@dataclass
class EICConfig:
    problem: str = "DAS-CMOP4"
    dim: int = 10
    seed: int = 0
    init_fe: int | None = None
    max_fe: int = 400
    pop_size: int = 100
    generations: int = 20
    mc_samples: int = 1000
    tau1: float = 0.5
    tau2: float = 0.7
    transition_rounds: int = 8

    def validate(self):
        self.problem = normalize_problem_name(self.problem)
        if self.init_fe is None:
            self.init_fe = 11 * self.dim - 1
        if self.dim < 3 or not 2 <= self.init_fe < self.max_fe:
            raise ValueError("Require dim >= 3 and 2 <= init_fe < max_fe.")
        if self.pop_size < 3 or self.generations < 2 or self.mc_samples < 1:
            raise ValueError("Require pop_size >= 3, generations >= 2 and mc_samples >= 1.")
        if not 0 < self.tau1 < self.tau2 < 1:
            raise ValueError("Require 0 < tau1 < tau2 < 1.")
        if self.transition_rounds < 2 or self.transition_rounds % 2:
            raise ValueError("transition_rounds must be an even integer >= 2.")


class StageController:
    # The paper's early check uses Tr new batches (3*Tr new evaluations).
    # Count these after LHS because the initial archive already exceeds 3*Tr.
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage = 1
        self.batch_min_cv = []
        self.batch_feasible = []
        self.early_check_done = False

    def observe(self, g):
        cv = np.maximum(g, 0).sum(axis=1)
        self.batch_min_cv.append(float(cv.min()))
        self.batch_feasible.append(bool(np.any(cv == 0)))

    def update(self, fe, has_constraints=True):
        if not has_constraints:
            return 1
        rounds = self.cfg.transition_rounds
        if self.stage == 1 and not self.early_check_done and len(self.batch_min_cv) >= rounds:
            self.early_check_done = True
            values = np.asarray(self.batch_min_cv[:rounds])
            change = values[rounds // 2:].sum() - values[:rounds // 2].sum()
            if change >= 0 and not any(self.batch_feasible[:rounds]):
                self.stage = 3
        if fe > self.cfg.tau2 * self.cfg.max_fe:
            self.stage = 3
        elif fe > self.cfg.tau1 * self.cfg.max_fe and self.stage == 1:
            self.stage = 2
        return self.stage


def _best_front_indices(y, g, constrained):
    indices = np.arange(len(y))
    if constrained and g.shape[1]:
        cv = np.maximum(g, 0).sum(axis=1)
        indices = indices[cv == cv.min()]
    nd = NonDominatedSorting().do(y[indices], only_non_dominated_front=True)
    return indices[nd]


def _pof_log(mean, std):
    if mean.shape[1] == 0:
        return np.zeros(len(mean))
    positive_std = std > 0
    z = -mean / np.where(positive_std, std, 1.0)
    terms = np.where(positive_std, norm.logcdf(z), np.where(mean <= 0, 0.0, -np.inf))
    return terms.sum(axis=1)


def _nd_a_scores(mean, archive):
    if len(archive) == 0:
        return np.zeros(len(mean))
    return np.array([min(_vector_angle(point, other) for other in archive) for point in mean])


def _nd_pbi_scores(mean, archive, refs):
    combined = np.vstack((archive, mean))
    minimum, maximum = combined.min(axis=0), combined.max(axis=0)
    candidates = _normalize_objectives(mean, minimum, maximum)
    if not len(archive):
        return -candidates.mean(axis=1)
    observed = _normalize_objectives(archive, minimum, maximum)
    association, _, pbi = _pbi_stats(observed, refs, theta=5.0)
    occupied = np.unique(association)
    # Candidate niches are restricted to directions occupied by the archive.
    niche, _, candidate_pbi = _pbi_stats(candidates, refs[occupied], theta=5.0)
    best = np.array([pbi[association == index].min() for index in occupied])
    return best[niche] - candidate_pbi


def _pd(values, direction):
    # Reuse PBI geometry, replacing its axial term with mean(F), Eq. (22).
    _, axial, pbi = _pbi_stats(values, direction[None, :], theta=5.0)
    return values.mean(axis=1) + pbi - axial


def _epdi_scores(mean, std, archive, refs, rng, mc_samples):
    direction = refs[int(rng.integers(len(refs)))]
    direction = direction / np.linalg.norm(direction)
    threshold = float(_pd(archive, direction).min())
    # Common normal draws compare candidates along the same random direction.
    normal_draws = rng.standard_normal((mc_samples, mean.shape[1]))
    scores = np.empty(len(mean))
    for i, point in enumerate(mean):
        samples = point + normal_draws * std[i]
        improvement = np.maximum(threshold - _pd(samples, direction), 0.0)
        scores[i] = improvement.mean()
    return scores


def select_eic_batch(archive_y, archive_g, mean, std, g_mean, g_std, refs,
                     stage, rng, mc_samples, batch_size=3):
    if not 1 <= batch_size <= 3:
        raise ValueError("EIC selects one to three points per batch.")
    if len(mean) < batch_size:
        raise ValueError("The candidate pool is smaller than the requested EIC batch.")
    available = np.arange(len(mean))
    temporary_y, temporary_g = archive_y.copy(), archive_g.copy()
    selected, members = [], []
    has_feasible = bool(feasible_mask(archive_g).any())
    constrained = stage != 1
    for member in range(batch_size):
        if member < 2:
            eligible = available[_best_front_indices(mean[available], g_mean[available], constrained)]
            observed = _best_front_indices(temporary_y, temporary_g, constrained)
            if member == 0:
                scores = _nd_a_scores(mean[eligible], temporary_y[observed])
                label = "nd_a_cdp" if constrained else "nd_a"
            else:
                scores = _nd_pbi_scores(mean[eligible], temporary_y[observed], refs)
                label = "nd_delta_pbi_cdp" if constrained else "nd_delta_pbi"
        else:
            eligible = available
            if constrained and not has_feasible:
                scores = _pof_log(g_mean[eligible], g_std[eligible])
                label = "pof"
            else:
                observed = temporary_y
                if constrained:
                    observed = temporary_y[feasible_mask(temporary_g)]
                scores = _epdi_scores(mean[eligible], std[eligible], observed, refs, rng, mc_samples)
                label = "epdi_pof" if constrained else "epdi"
                if constrained:
                    with np.errstate(divide="ignore"):
                        scores = np.log(scores) + _pof_log(g_mean[eligible], g_std[eligible])
        chosen = int(eligible[int(np.argmax(scores))])
        selected.append(chosen)
        members.append(label)
        available = available[available != chosen]
        temporary_y = np.vstack((temporary_y, mean[chosen]))
        temporary_g = np.vstack((temporary_g, g_mean[chosen]))
    return np.asarray(selected, dtype=int), members


class StageObjectives:
    def __init__(self, objectives, constraints, active_constraint):
        self.objectives = objectives
        self.constraints = constraints
        self.active_constraint = active_constraint

    def predict_mean(self, x):
        y = self.objectives.predict_mean(x)
        if self.active_constraint is None:
            return y
        violation = np.maximum(self.constraints.predict_mean(x)[:, self.active_constraint], 0)
        return np.column_stack((y, violation))


def generate_candidates(problem, archive_x, archive_y, archive_g, objective_gp,
                        constraint_gp, stage, cfg, refs, seed):
    active = None
    stage_y = archive_y
    if stage == 2:
        active = int(np.argmin((archive_g <= 0).mean(axis=0)))
        stage_y = np.column_stack((archive_y, np.maximum(archive_g[:, active], 0)))
    proxy = Problem(n_var=problem.dim, n_obj=stage_y.shape[1],
                    n_ieq_constr=archive_g.shape[1] if stage == 3 else 0,
                    xl=problem.pymoo_problem.xl, xu=problem.pymoo_problem.xu)
    values = {"X": archive_x, "F": stage_y}
    if stage == 3:
        values["G"] = archive_g
    population = Population.new(**values)
    rng = np.random.default_rng(seed)
    survival = ReferenceDirectionSurvival(refs)
    initial = survival.do(proxy, population, n_survive=min(cfg.pop_size, len(population)), random_state=rng)
    result = solve(
        "nsga3", problem=proxy, archive_x=archive_x,
        surrogate=StageObjectives(objective_gp, constraint_gp, active),
        constraint_surrogate=constraint_gp if stage == 3 else None,
        pop_size=cfg.pop_size, saea_steps=cfg.generations, seed=seed,
        seed_x=initial.get("X"), ref_dirs=refs,
        crossover=SBX(prob=1.0, eta=20), mutation=PM(prob=1.0, prob_var=1.0 / problem.dim, eta=20),
    )
    return result.x, active


def _metrics(y, g, ref_point, reference):
    feasible = y[feasible_mask(g)]
    if not len(feasible):
        return 0.0, None, np.empty((0, y.shape[1]))
    return hypervolume(feasible, ref_point), float(cKDTree(feasible).query(reference)[0].mean()), pareto_front(feasible)


def run_eic_mssaea(cfg, logger=print, initial_archive=None):
    cfg.validate()
    started = time.perf_counter()
    problem = make_problem(cfg.problem, cfg.dim)
    if initial_archive is None:
        x = latin_hypercube_sample(n_samples=cfg.init_fe, dim=cfg.dim,
                                  lower=problem.lower, upper=problem.upper, seed=cfg.seed)
        y, g = evaluate_objectives_and_constraints(problem, x)
    else:
        x, y, g = (np.asarray(value, dtype=np.float64).copy() for value in initial_archive)
        if x.shape != (cfg.init_fe, cfg.dim) or y.shape != (cfg.init_fe, problem.n_obj) or g.ndim != 2 or len(g) != cfg.init_fe:
            raise ValueError("Initial archive dimensions must match the configuration and problem.")
    reference = reference_front(cfg.problem, cfg.dim)
    ref_point = get_reference_point(y)
    controller = StageController(cfg)
    rng = np.random.default_rng(cfg.seed)
    directions = {n: _build_reference_directions(n, cfg.pop_size, cfg.seed)
                  for n in (problem.n_obj, problem.n_obj + 1)}
    for refs in directions.values():
        refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    hv, igd, front = _metrics(y, g, ref_point, reference)
    fe_history, hv_history, igd_history = [len(x)], [hv], [igd]
    history = []
    logger(f"eic_mssaea | problem = {cfg.problem} | dim = {cfg.dim} | seed = {cfg.seed}")
    logger(f"FE = {len(x)} | HV = {hv:.12f} | IGD = {igd if igd is not None else 'NA'}")
    batch = 0
    while len(x) < cfg.max_fe:
        stage = controller.update(len(x), has_constraints=g.shape[1] > 0)
        objective_gp = fit_gp_surrogates(archive_x=x, archive_y=y, seed=cfg.seed + batch)
        constraint_gp = None
        if g.shape[1]:
            constraint_gp = fit_gp_surrogates(archive_x=x, archive_y=g, seed=cfg.seed + batch)
        stage_refs = directions[problem.n_obj + (stage == 2)]
        candidates, active = generate_candidates(problem, x, y, g, objective_gp, constraint_gp,
                                                 stage, cfg, stage_refs, cfg.seed + batch)
        mean, std = objective_gp.predict_mean(candidates), objective_gp.predict_std(candidates)
        g_mean = np.empty((len(candidates), 0))
        g_std = np.empty_like(g_mean)
        if constraint_gp is not None:
            g_mean, g_std = constraint_gp.predict_mean(candidates), constraint_gp.predict_std(candidates)
        indices, members = select_eic_batch(
            y, g, mean, std, g_mean, g_std, directions[problem.n_obj], stage, rng,
            cfg.mc_samples, batch_size=min(3, cfg.max_fe - len(x)),
        )
        new_x = candidates[indices]
        new_y, new_g = evaluate_objectives_and_constraints(problem, new_x)
        controller.observe(new_g)
        batch += 1
        # Only true evaluations enter the permanent archive.
        for index, member in enumerate(members):
            x = np.vstack((x, new_x[index]))
            y = np.vstack((y, new_y[index]))
            g = np.vstack((g, new_g[index]))
            hv, igd, front = _metrics(y, g, ref_point, reference)
            fe_history.append(len(x))
            hv_history.append(hv)
            igd_history.append(igd)
            history.append(dict(fe=len(x), batch=batch, stage=stage, criterion=member,
                                active_constraint=active, x=new_x[index].tolist(),
                                y=new_y[index].tolist(), g=new_g[index].tolist(), hv=hv, igd=igd))
        logger(f"batch {batch} | stage = {stage} | FE = {len(x)} | front = {len(front)} | "
               f"HV = {hv:.12f} | IGD = {igd if igd is not None else 'NA'}")
    return dict(method="eic_mssaea", config=asdict(cfg), seed=cfg.seed,
                reference_point=ref_point.tolist(), reference_front=reference.tolist(),
                final_hv=hv, final_igd=igd, final_front=front.tolist(),
                archive_x=x.tolist(), archive_f=y.tolist(), archive_g=g.tolist(),
                fe_history=fe_history, hv_history=hv_history, igd_history=igd_history,
                history=history, wall_clock_sec=time.perf_counter() - started,
                stage_transition_check="after transition_rounds post-LHS batches")


def parse_args():
    parser = argparse.ArgumentParser(description="EIC-MSSAEA using the repository GP and NSGA-III solver.")
    parser.add_argument("--problem", type=normalize_problem_name, choices=SUPPORTED_PROBLEMS, default="DAS-CMOP4")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_fe", type=int, default=None, help="Defaults to 11*dim-1.")
    parser.add_argument("--max_fe", type=int, default=400)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--mc_samples", type=int, default=1000)
    parser.add_argument("--tau1", type=float, default=.5)
    parser.add_argument("--tau2", type=float, default=.7)
    parser.add_argument("--transition_rounds", type=int, default=8)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--log_path", type=Path)
    args = parser.parse_args()
    config = EICConfig(**{name: getattr(args, name) for name in EICConfig.__dataclass_fields__})
    config.validate()
    return args, config


def main():
    args, cfg = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_path or ROOT_DIR / "testing_logs" / cfg.problem / f"eic_mssaea_{cfg.problem.lower()}_seed{cfg.seed}_{stamp}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        def logger(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()
        summary = run_eic_mssaea(cfg, logger=logger)
    output = args.output_json or log_path.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
