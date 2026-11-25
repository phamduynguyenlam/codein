# CoDeIn: Constrained Deep Infill Sampling in Expensive CMOPs

CoDeIn learns to grade archive points as seeds for constrained NSGA-II and select offspring for black-box evaluation. Each offspring embedding is concatenated with the masked mean archive embedding and passed through a shared MLP before the dueling Q decoder. This architecture requires newly trained checkpoints.

## Setup

Python 3.11 is recommended. Run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Training

`--problem` identifies the held-out family.

| Flag | Training benchmarks |
| --- | --- |
| `--problem MW` | DAS-CMOP1?9, 8 decision variables, difficulty 1 |
| `--problem DAS` | MW1?14, 8 decision variables |

```powershell
python train.py --problem MW --seed 100000 --epoch 50 --num_workers 9 --device cpu
```

Training and evaluation default to 300 FE, with 100 initial LHS points and 200 infill evaluations, up to five evaluations per candidate pool. Use `--init_fe`, `--max_fe`, and `--batch` to override these values. Training options also include `--parents`, `--temperature`, `--value_coef`, `--q_coef`, and `--reward`. Checkpoints are saved under `training_logs/codein/codein_train_<held-out-family>_set1_<timestamp>/`.

The default `--reward pareto` scores relative improvement in the best scaled constraint violation until the first feasible evaluation, which earns 1. With a feasible archive, an infeasible point or a dominated feasible point earns −1. A nondominated feasible point earns the Pareto improvement score using Euclidean distances to the feasible front. `--reward hv` uses the same feasibility phase and normalized HV improvement for feasible evaluations thereafter.

## Evaluation

Select a benchmark instance. Both families default to 10 decision variables at evaluation time. Train and test accept `--dim` and `--solver` overrides. NSGA-II is the default, and `--solver moead_ego` is available. Tchebycheff PI uses MOEA/D-EGO for its candidate weight vectors. The tester resolves the latest best-reward checkpoint for the corresponding held-out family and selected `--reward`. Use `--agent_pth` to specify a checkpoint directly.

```powershell
python test.py --problem MW1 --seeds 0 1 2 --infill codein --compare_infill random
python test.py --problem DAS-CMOP7 --seed 7 --infill codein --policy_mode softmax_sample --temperature 1.0
```

`--seed` runs one seed. With neither seed option the tester chooses one seed automatically.

CodeIn samples each infill candidate from softmax over the valid candidates' Q-values divided by `--temperature`. Training and evaluation both use this by default. Evaluation also accepts `--policy_mode q_greedy` for argmax.

Each seed uses the same initial LHS for all compared policies. Its HV reference is 1.1 times the maximum observed initial value of each objective. HV and IGD use the feasible nondominated archive. With zero feasible solutions the log reports HV as zero and IGD as `NA`. The JSON records missing IGD as `null` and includes the valid-seed count at each FE.

The result JSON stores the reference front, each seed?s reference point and evaluation history, and the objective and constraint archives in evaluation order. PNG and PDF figures show mean ? population standard deviation for HV and IGD. IGD averages include only seeds with a defined value. DAS-CMOP reference fronts are deterministic samples generated for the selected dimension. MW uses the reference fronts provided by pymoo, which may fetch benchmark data on first use.

## Quick verification

```powershell
python test.py --problem DAS-CMOP1 --seeds 0 1 --init_fe 6 --max_fe 8 --offspring_size 5 --random_model
```

## Problem interface

Each problem returns objective and constraint matrices `(F, G)` through `evaluate_with_constraints(x)`. Feasibility means `G <= 0` for every constraint. An unconstrained problem uses `G` with shape `[N, 0]`. CodeIn fixes each constraint scale to the maximum absolute initial LHS value, floored at `1e-8`.

## EIC-MSSAEA baseline

```powershell
python baseline/eic_mssaea.py --problem DAS-CMOP4 --init_fe 100 --max_fe 300 --generations 15 --seed 0
```

This standalone baseline uses the repository GP, the NSGA-III solver and shared infill geometry. It implements unconstrained objective search, single-constraint violation as an additional objective, and full constrained search, followed by serial ND-A, ND-delta-PBI and EPDI selection with CDP/PoF in stages 2 and 3. Each batch evaluates up to three distinct points. The final batch respects the remaining evaluation budget.

Defaults follow Table I of Wu et al., *A Multistage Expensive Constrained Multiobjective Optimization Algorithm Based on Ensemble Infill Criterion*, DOI 10.1109/TEVC.2024.3400832, except for the requested use of the existing GP. Defaults are `init_fe=11*dim-1`, `max_fe=400`, `pop_size=100`, `generations=20`, `mc_samples=1000`, `tau1=0.5`, `tau2=0.7` and `transition_rounds=8`. The early stage-transition check counts the first `transition_rounds` batches after LHS, resolving the paper's ambiguous `FE=3*Tr` condition. Reference directions and variation operators are supplied to the shared solver for this baseline.

Runs save a text log and JSON under `testing_logs/<problem>/`, including stage, selected criterion, full archives, HV and IGD histories. IGD uses all accumulated feasible evaluations, following the current repository metric. Figures are omitted. Use `--output_json` and `--log_path` to choose destinations.
