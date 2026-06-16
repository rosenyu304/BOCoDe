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
python -m algorithms.single_obj.vanilla_highdim_bo --problem Car --init 10 --iters 50
python -m algorithms.single_obj.turbo --dataset CrossedBarrel --init 10 --iters 40
python -m algorithms.single_obj_constrained.scbo --problem PressureVessel --iters 80
python -m algorithms.multi_obj.qnehvi --problem Penicillin --init 10 --iters 50
```

Common flags (every script):

- `--seed` (default **42**) — seeds the RNGs, including the initial sample. The
  10-seed experiment uses 0, 10, 20, …, 90.
- `--show_progress` — print the best-so-far value (with wall-clock time) at each
  iteration, then the summary line at the bottom.
- `--saved_full_experiment [PATH]` — save the full per-iteration trace to a NumPy
  `.npz` (default name `<algorithm>_<problem>_seed<seed>.npz`) with keys: `seed`,
  `acquisition_function`, `best`, `iterations` (0…n, where 0 is the initial design),
  `per_iteration_value` (best-so-far), `wall_time` (seconds from trial start, 0 at
  iteration 0), `mean` and `variance` (GP posterior at the chosen point; `nan` for
  random search and at iteration 0), and `per_iteration_acquisition_function_value`.

| Folder | Implemented | Notes |
|---|---|---|
| `single_obj/` | ✅ `random_search`, `single_task_gp`, `vanilla_highdim_bo`, `turbo`, `standard_gp` | problem + dataset variants |
| `single_obj_constrained/` | ✅ `random_search`, `constrained_ei`, `scbo` | problem variant (no constrained dataset problems yet) |
| `multi_obj/` | ✅ `random_search`, `qnehvi`, `qnparego` | problem variant; hypervolume-tracked |
| `multi_obj_constrained/` | ✅ `constrained_qnehvi`, `constrained_qparego` | problem variant |
| `single_obj_mixed_variable/` | ⏳ next | dedicated mixed-variable BO (current baselines already handle mixed problems via rounding) |

Dataset-optimization variants are provided for the single-objective unconstrained
algorithms, since BoCoDe's discrete dataset problems (the PV-Lab materials sets)
are single-objective and unconstrained.

### TFM (transformer-foundation-model) algorithms

These use a pretrained TabPFN surrogate instead of a GP and need a **separate
environment** (the TabPFN fork pins scikit-learn and Python <3.12) — see
[docs/tfm_setup.md](../docs/tfm_setup.md).

| Algorithm | Notes |
|---|---|
| `single_obj/git_bo.py` | GIT-BO with a gradient-informed active subspace; `--rank 5` (fixed, original) or `--rank marzouk` (Zahm–Marzouk certified rank) |
| `single_obj_constrained/pfn_cei.py` | constrained EI with the TabPFN regressor; objective + all constraints scored in one parallel forward |

The rank-selection logic is pure NumPy and is unit-tested (`tests/test_tfm.py`);
the TabPFN runs are skipped unless the fork is installed.

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
