---
name: bocode
description: Benchmark an optimization algorithm on the BOCoDe suite (307 black-box problems - engineering, HPO, synthetic). Use when the user wants to test, compare, or debug a Bayesian-optimization or evolutionary method on BOCoDe problems, or to pick problems by class/domain/dimension.
---

# Benchmarking an optimizer on BOCoDe

## Conventions (non-negotiable)

- All objectives are **maximized**. A minimizing optimizer must negate.
- Constraints: `evaluate()` returns `(objectives, constraints)`; a point is
  feasible when every constraint value is `<= 0`.
- Inputs are `torch.Tensor` of shape `(batch, dim)` within `problem.bounds`.
  Use `torch.float64` for GP-based methods.

## 1. Pick problems

```python
import bocode
names = bocode.list_problems(application="Engineering", constrained=False)
meta  = bocode.get_metadata(names[0])   # dim, num_objectives, num_constraints,
                                        # input_type, source citation
```

Filter kwargs: `application` ("Engineering"/"HPO"/...), `num_objectives`,
`constrained`, `input_type` ("continuous"/"discrete"/"mixed"), `scalable`.
Synthetic test functions are separate: `bocode.list_synthetic()`.
Check `meta["dim"]` before committing to expensive methods (e.g.
`LassoLeukemia` is 7129-D, `AntPolicySearchProblem` 840-D).

## 2. Run the user's optimizer

Minimal loop against any problem:

```python
problem = bocode.get_problem("Car")()
X = problem.sample(n_init, seed=seed)          # respects mixed-variable dims
Y, G = problem.evaluate(X)                     # maximize Y; feasible G <= 0
for _ in range(iters):
    x_next = user_optimizer.propose(X, Y, G, problem.bounds)
    y, g = problem.evaluate(x_next)
    # append and continue; track best FEASIBLE y so far
```

## 3. Compare against reference baselines

`algorithms/` (repo checkout, not in the wheel) has single-file baselines with
a common CLI — same seeds, same accounting:

```bash
python -m algorithms.single_obj.gp_ucb --problem Car --init 10 --iters 40 --seed 0
```

Folders by class: `single_obj`, `single_obj_constrained`, `multi_obj`,
`multi_obj_constrained`. `--saved_full_experiment` writes a per-iteration
`.npz` trace (keys: `best`, `per_iteration_value`, `wall_time`, ...).

## 4. Report

- Run multiple seeds (paper protocol: 25 seeds) and report the median
  best-so-far curve, not a single run.
- For constrained problems also report the feasibility rate (fraction of runs
  reaching any feasible point within budget).
- For multi-objective problems track hypervolume (`algorithms/_hv_utils.py`).
- Rank across problems with mean arithmetic rank per iteration;
  Friedman + Nemenyi critical difference for significance claims.

## Pitfalls

- Wrong sign: if the user's method "gets worse", check they are maximizing.
- Do not evaluate outside `problem.bounds`; use `problem.sample()` for valid
  initial designs (it handles integer/categorical dims).
- Some problems need optional extras (`pip install "bocode[mujoco]"` etc.) or
  a one-time data download (cached in `~/.cache/bocode`); the ImportError
  message names the fix.
