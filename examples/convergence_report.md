# Convergence validation report

Budget: 10 initial + 30 iterations, seeds [0, 1, 2]. `best` is the best objective found (minimization sense), mean +/- std across seeds; `gap` = best - f_opt; `vs random` = random_mean - algo_mean (positive is better than random).

## Branin  (f_opt = None)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | 1.737 ± 0.813 | — | — | baseline |
| single_task_gp | 0.40936 ± 0.00712 | — | +1.328 | beats random |
| standard_gp | 0.40032 ± 0.00173 | — | +1.337 | beats random |
| vanilla_highdim_bo | 0.40059 ± 0.0016 | — | +1.336 | beats random |
| turbo | 0.40349 ± 0.00367 | — | +1.334 | beats random |

## Sellar  (f_opt = 3.18339)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | 12.328 ± 1.09 | 9.145 | — | baseline |
| constrained_ei | 3.1952 ± 0.0103 | 0.01177 | +9.133 | converges |
| scbo | 3.3064 ± 0.0729 | 0.1231 | +9.022 | converges |

## Allison  (f_opt = 0.5698)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | 0.94954 ± 0.144 | 0.3797 | — | baseline |
| single_task_gp | 0.60448 ± 0.035 | 0.03468 | +0.3451 | beats random |
| standard_gp | 0.59934 ± 0.0302 | 0.02954 | +0.3502 | beats random |
| vanilla_highdim_bo | 0.6414 ± 0.0635 | 0.0716 | +0.3081 | beats random |
| turbo | 0.62758 ± 0.0446 | 0.05778 | +0.322 | beats random |

## MiniAeroWing  (f_opt = 242.27)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | 269.54 ± 12.9 | 27.27 | — | baseline |
| single_task_gp | 262.13 ± 8.21 | 19.86 | +7.407 | beats random |
| standard_gp | 264.75 ± 8.94 | 22.48 | +4.786 | beats random |
| vanilla_highdim_bo | 271.87 ± 13 | 29.6 | -2.336 | no better than random |
| turbo | 244.95 ± 1.94 | 2.676 | +24.59 | converges |

## PEARL  (f_opt = 585.3)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | no feasible point | — | — | no feasible found |
| constrained_ei | 8040.1 ± 0 (2/3 infeasible) | 7455 | — | no better than random |
| scbo | no feasible point | — | — | no feasible found |

## PressureVessel  (f_opt = None)

| algorithm | best (mean ± std) | gap | vs random | verdict |
|---|---|---|---|---|
| random_search | 7.7028e+05 ± 0 | — | — | baseline |
| constrained_ei | 7.7028e+05 ± 0 | — | +0 | no better than random |
| scbo | 7.7028e+05 ± 0 | — | +0 | no better than random |

