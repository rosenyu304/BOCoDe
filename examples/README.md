# Running BoCoDe experiments

How to run the Bayesian-optimization baselines in `algorithms/` against the BoCoDe
problem suite — one algorithm on one problem, a full batch, or a 100+-problem sweep
on a cluster. Every run can save its complete per-iteration trace to a `.npz`
("experimental mode"), so results are reproducible and analyzable offline.

- `example.ipynb` — a notebook demo: one cell per algorithm, each run in
  experimental mode (saving the full trace), plus a cell that loads a saved `.npz`.
- `run_experiments.py` — the batch runner: `(algorithm × problem × seed)`, saves
  every trace and a summary CSV, computes the gap to the known optimum where one is
  recorded, and shards for cluster array jobs.
- `convergence_validation.py` — the validation harness: runs the single-objective
  baselines against problems with a known optimum over several seeds and reports the
  mean ± std gap to the optimum and the improvement over random search, so you can
  certify which algorithm/problem combinations actually *converge*.

## 1. Set up the environment (mamba)

BoCoDe targets Python ≥3.10 (CI uses 3.12). Create a dedicated env and install the
package editable with the dependency groups/extras you need:

```bash
mamba create -n bocode python=3.12 -y
mamba activate bocode

# core only:
pip install -e .

# core + everything (all problem extras + dev tooling) — recommended for a full sweep:
pip install -e ".[all]" --group dev
```

Individual problem extras (install only what you run): `mujoco`, `control`,
`modact`, `neorl`, `box2d`, `truss`, `mazda`, `hpo`, `viz`. A problem that needs a
missing extra raises an actionable `ImportError` and is skipped by the batch runner.

The **TFM** algorithms (GIT-BO, PFN-CEI) need a *separate* TabPFN environment — the
fork pins scikit-learn and Python <3.12. See `../docs/tfm_setup.md`.

## 2. Run a single algorithm

Every algorithm is a runnable module with a CLI. `--saved_full_experiment` turns on
experimental mode (writes `<algorithm>_<problem>_seed<seed>.npz`); `--show_progress`
prints best-so-far each iteration:

```bash
python -m algorithms.single_obj.vanilla_highdim_bo --problem Branin --init 10 --iters 50 \
    --seed 0 --show_progress --saved_full_experiment
python -m algorithms.single_obj_constrained.scbo --problem PressureVessel --iters 80 \
    --saved_full_experiment results/scbo_pv.npz
```

## 3. Batch over many problems × algorithms × seeds

`run_experiments.py` pairs each problem with the compatible algorithms (decided from
its metadata: single/multi-objective, constrained/unconstrained), runs every seed,
saves each trace, and writes `summary_task<id>.csv`:

```bash
# a few problems, all compatible algorithms, three seeds:
python examples/run_experiments.py --problems Branin Sellar PressureVessel \
    --seeds 0 10 20 --n-init 10 --iters 50 --outdir results

# the full suite (every registered problem), five seeds:
python examples/run_experiments.py --problems all --seeds 0 10 20 30 40 \
    --n-init 20 --iters 100 --outdir results

# one algorithm on a filtered slice — e.g. SingleTaskGP on every single-objective,
# unconstrained, continuous problem:
python examples/run_experiments.py --problems all \
    --objectives 1 --unconstrained --input-type continuous \
    --algorithms single_task_gp --seeds 0 10 20
```

`--algorithms` restricts to specific algorithm labels (default = the standard
per-category baselines); it also unlocks the opt-in mixed-variable methods
(`single_task_gp_mixed`, `random_search_mixed`) on single-objective problems. When
`--problems all`, `--objectives/--constrained/--unconstrained/--input-type` narrow
the set via `bocode.list_problems(...)`.

Each line of the summary CSV is `algorithm,problem,seed,best,gap,npz`, where `gap`
is `best_found − f_opt` for single-objective problems whose optimum is recorded in
the metadata (otherwise blank). `best` is in the maximization convention the
algorithms use (`−objective` for single-objective, hypervolume for multi-objective).

## 4. Scale to a cluster (SLURM array)

The full grid is large (100+ problems × ~3–5 algorithms × seeds). Split it across an
array job with `--task-id` / `--num-tasks`: shard `k` runs jobs `k, k+N, k+2N, …`, so
`N` tasks cover the grid with no overlap.

```bash
#!/bin/bash
#SBATCH --job-name=bocode-sweep
#SBATCH --array=0-63              # 64 shards
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%a.out

source ~/miniforge3/etc/profile.d/conda.sh
mamba activate bocode

python examples/run_experiments.py \
    --problems all --seeds 0 10 20 30 40 \
    --n-init 20 --iters 100 \
    --outdir results \
    --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"
```

Submit with `sbatch sweep.slurm`. Each shard writes its own `summary_task<id>.csv`
and its own `.npz` files into the shared `results/` directory; concatenate the
summaries afterwards (`cat results/summary_task*.csv`). Pick `--num-tasks` so each
shard finishes inside the wall-clock limit (more shards = shorter each).

## 5. Load a saved run

```python
import numpy as np
d = np.load("results/vanilla_highdim_bo_Branin_seed0.npz")
print(d.files)                      # seed, acquisition_function, best, iterations,
                                    # per_iteration_value, wall_time, mean, variance, ...
print(d["per_iteration_value"])     # best-so-far per iteration (0 = initial design)
```

## 6. Validate convergence (not just "it runs")

`convergence_validation.py` runs the single-objective baselines against problems
whose optimum is known, over several seeds, and writes `convergence_report.md` (and
`.csv`) summarizing, per `(problem, algorithm)`, the mean ± std best objective, the
gap to the optimum, and the improvement over random search:

```bash
python examples/convergence_validation.py --seeds 0 1 2 3 4 --iters 40
# override the problem set (default: Branin, Sellar, Allison, MiniAeroWing, PEARL, PressureVessel):
python examples/convergence_validation.py --problems Sellar MiniAeroWing --seeds 0 1 2
```

Use it to certify that an algorithm actually converges on a problem (the unit tests
only check that runs are monotone on tiny budgets). A committed `convergence_report.md`
shows an example result.

