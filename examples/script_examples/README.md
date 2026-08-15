# Per-category experiment scripts

One script per problem category from `CATEGORIZATION.md`. Each runs the standard
baselines on all problems in that category, CPU-capped at 12 threads. A commented
SLURM-array block is included for cluster use.

```bash
bash examples/script_examples/example_run_experiment_single_obj_unconstrained_continuous.sh
```

Results (per-run `.npz` + `summary_task0.csv`) land in `examples/results/`.
