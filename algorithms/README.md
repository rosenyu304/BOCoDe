# BoCoDe Algorithms

Single-file, CleanRL-style reference implementations of Bayesian-optimization
baselines that run against the BoCoDe problem suite. These are **research
scripts**, not part of the installable `bocode` package: each file is
self-contained, seeded, and runnable on its own.

## Layout and status

Folders mirror the problem categories the algorithm targets. Each script exposes
`optimize_problem(problem, ...)` and (where a discrete candidate set is the natural
search space) `optimize_dataset(dataset_problem, ...)`, plus a CLI:

```bash
python -m algorithms.single_obj.vanilla_bo --problem Car --init 10 --iters 50
python -m algorithms.single_obj.turbo --dataset CrossedBarrel --init 10 --iters 40
python -m algorithms.single_obj_constrained.scbo --problem PressureVessel --iters 80
python -m algorithms.multi_obj.qnehvi --problem Penicillin --init 10 --iters 50
```

| Folder | Implemented | Notes |
|---|---|---|
| `single_obj/` | ✅ `random_search`, `vanilla_bo`, `turbo`, `standard_gp` | problem + dataset variants |
| `single_obj_constrained/` | ✅ `random_search`, `constrained_ei`, `scbo` | problem variant (no constrained dataset problems yet) |
| `multi_obj/` | ✅ `random_search`, `qnehvi`, `qnparego` | problem variant; hypervolume-tracked |
| `multi_obj_constrained/` | ✅ `random_search` (via `multi_obj`), `constrained_qnehvi` | problem variant |
| `single_obj_mixed_variable/` | ⏳ Push 3 | mixed-variable BO (with the firefly mixed-integer problems) |

Dataset-optimization variants are provided for the single-objective unconstrained
algorithms, since BoCoDe's discrete dataset problems (the PV-Lab materials sets)
are single-objective and unconstrained.

## Conventions

- **One algorithm per script.** Every step lives in the file; the shared
  `_bo_utils.py` holds only the evaluation plumbing (seeding, the problem/dataset
  adapters, GP fitting, the result trace).
- **Maximization.** Every adapter exposes an objective to *maximize* (BoCoDe's
  convention); the continuous search space is the unit cube `[0, 1]^d`, scaled to
  the problem bounds before evaluation.
- **Reproducibility**: all RNGs are seeded and PyTorch is made deterministic.
- **Always a Random search baseline** per category.

## Correctness notes carried from review

For the Vanilla BO baseline (Hvarfner et al., 2024), the implementation must use
`LogExpectedImprovement` (not `ExpectedImprovement`), `raw_samples=512` with
restarts from the best 4 candidates, and `sample_around_best=True`; and the
`best_f` passed to the acquisition must be in the same (normalized) space as the
GP training targets. See `Research_Plan.md` §10 for the full list.
