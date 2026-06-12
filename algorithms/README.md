# BoCoDe Algorithms

Single-file, CleanRL-style reference implementations of Bayesian-optimization
baselines that run against the BoCoDe problem suite. These are **research
scripts**, not part of the installable `bocode` package: each file is
self-contained, seeded, and runnable on its own.

> Status: **scaffolding only** (Foundation push, dev/2026_06). The folders below
> define where each algorithm will live; the implementations land in Push 2.

## Layout

Folders mirror the problem categories the algorithm targets:

| Folder | Target problems | Planned baselines |
|---|---|---|
| `single_obj/` | single-objective, unconstrained, continuous | Random search, TuRBO, Vanilla BO, Standard GP (HDBO) |
| `single_obj_constrained/` | single-objective, constrained, continuous | Random search, SCBO, constrained EI |
| `single_obj_mixed_variable/` | single-objective, mixed integer/categorical | Random search, mixed-variable BO (Bounce / MCBO-style) |
| `multi_obj/` | multi-objective, unconstrained | Random search, qEHVI, qNEHVI, qNParEGO |
| `multi_obj_constrained/` | multi-objective, constrained | Random search, constrained qNEHVI, qParEGO |

## Conventions (for Push 2 implementations)

- **One algorithm per script.** Every training detail lives in the file; a
  separate evaluation harness handles metrics and plotting.
- **Two entry points per algorithm**: a *problem-optimization* loop (query the
  problem's `evaluate`) and a *dataset-optimization* loop (select from a fixed
  candidate set, as in the PV-Lab benchmarking framework).
- **Reproducibility**: seed all RNGs, make PyTorch deterministic, log
  hyperparameters.
- **Always include a Random search baseline** for each category.

## Correctness notes carried from review

For the Vanilla BO baseline (Hvarfner et al., 2024), the implementation must use
`LogExpectedImprovement` (not `ExpectedImprovement`), `raw_samples=512` with
restarts from the best 4 candidates, and `sample_around_best=True`; and the
`best_f` passed to the acquisition must be in the same (normalized) space as the
GP training targets. See `Research_Plan.md` §10 for the full list.
