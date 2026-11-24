# CodeIn

CodeIn learns to grade archive points as seeds for MOEA/D-EGO and select offspring for black-box evaluation.

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
| `--problem MW` | DAS-CMOP1?9, 10 decision variables, difficulty 1 |
| `--problem DAS` | MW1?14, 15 decision variables |

```powershell
python train.py --problem MW --seed 100000 --epoch 50 --num_workers 9 --device cpu
```

Training and evaluation default to 300 FE, with 100 initial LHS points and 200 infill evaluations, up to five evaluations per candidate pool. Use `--init_fe`, `--max_fe`, and `--batch` to override these values. Training options also include `--num_seeds`, `--temperature`, `--value_coef`, `--q_coef`, and `--reward_scheme`. Checkpoints are saved under `training_logs/codein/codein_train_<held-out-family>_set1_<timestamp>/`.

## Evaluation

Select a benchmark instance. The tester resolves the latest best-reward checkpoint for the corresponding held-out family. Use `--agent_pth` to specify a checkpoint directly.

```powershell
python test.py --problem MW1 --seeds 0 1 2 --infill codein --compare_infill random
python test.py --problem DAS-CMOP7 --seed 7 --infill codein --policy_mode softmax_sample --temperature 1.0
```

`--seed` runs one seed. With neither seed option the tester chooses one seed automatically.

CodeIn samples each infill candidate from softmax over the valid candidates' Q-values divided by `--temperature`. Training and evaluation both use this by default. Evaluation also accepts `--policy_mode q_greedy` for argmax.

Each seed uses the same initial LHS for all compared policies. Its HV reference is 1.1 times the maximum observed initial value of each objective. HV and IGD use the feasible nondominated archive. With zero feasible solutions the log reports HV as zero and IGD as `NA`. The JSON records missing IGD as `null` and includes the valid-seed count at each FE.

The result JSON stores the reference front, each seed?s reference point and evaluation history, and the objective and constraint archives in evaluation order. PNG and PDF figures show mean ? population standard deviation for HV and IGD. IGD averages include only seeds with a defined value. DAS-CMOP reference fronts are deterministic samples generated for the 10-variable instances. MW uses the reference fronts provided by pymoo, which may fetch benchmark data on first use.

## Quick verification

```powershell
python test.py --problem DAS-CMOP1 --seeds 0 1 --init_fe 6 --max_fe 8 --offspring_size 5 --random_model
```

## Problem interface

Each problem returns objective and constraint matrices `(F, G)` through `evaluate_with_constraints(x)`. Feasibility means `G <= 0` for every constraint. An unconstrained problem uses `G` with shape `[N, 0]`. CodeIn fixes each constraint scale to the maximum absolute initial LHS value, floored at `1e-8`.
