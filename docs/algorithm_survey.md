# BO algorithm survey & roadmap

A survey of recent (≈last decade, ICML/ICLR/NeurIPS/UAI) Bayesian-optimization
algorithms for each BoCoDe algorithm category, with what is implemented now and what is
deferred (and why). "BoTorch-native" = built on BoTorch primitives with no extra heavy
dependency; "port" = reimplemented from the authors' reference; "extra" = needs an
optional dependency.

Every algorithm is a single file under `algorithms/<category>/` exposing
`optimize_problem(problem, n_init, iters, seed) -> Result` and is auto-discovered by
`examples/run_experiments.py` (`_ALGOS`).

## Single-objective, unconstrained (continuous / high-dim)
| Algorithm | Venue | Status | Notes |
|---|---|---|---|
| Random search | — | ✅ implemented | baseline |
| Single-task GP (LogEI) | — | ✅ implemented | `single_task_gp` |
| Standard GP HDBO | Xu et al. 2024 | ✅ implemented | `standard_gp` |
| Vanilla high-dim BO | Hvarfner et al. 2024 | ✅ implemented | `vanilla_highdim_bo` |
| TuRBO | Eriksson et al., NeurIPS 2019 | ✅ implemented | `turbo` |
| **BAxUS** | **Papenmeier et al., NeurIPS 2022** | ✅ **implemented** | `baxus` (nested random subspaces + trust region; BoTorch-native port) |
| SAASBO | Eriksson & Jankowiak, UAI 2021 | ⛔ deferred | fully-Bayesian GP needs `jax`/`pyro` (not in the core env) — add as an extra |

## Single-objective, constrained
| Algorithm | Venue | Status | Notes |
|---|---|---|---|
| Random search | — | ✅ implemented | |
| Constrained EI (LogCEI) | Gardner et al. 2014 | ✅ implemented | `constrained_ei` |
| SCBO | Eriksson & Poloczek, AISTATS 2021 | ✅ implemented | `scbo` |

## Multi-objective (constrained / unconstrained)
| Algorithm | Venue | Status | Notes |
|---|---|---|---|
| Random search | — | ✅ implemented | |
| qNEHVI | Daulton et al., NeurIPS 2021 | ✅ implemented | `qnehvi` (+ constrained variant) |
| qNParEGO | Daulton et al. 2020 | ✅ implemented | `qnparego` (+ constrained variant) |
| **MESMO** (max-value entropy) | **Belakaria et al., NeurIPS 2019** | ✅ **implemented** | `mesmo` (BoTorch `qLowerBoundMultiObjectiveMaxValueEntropySearch`) |
| JES (joint entropy search) | Tu et al., NeurIPS 2022 | 🔜 easy follow-up | BoTorch-native (`qLowerBoundMultiObjectiveJointEntropySearch`); same scaffold as MESMO |
| MORBO | Daulton et al., ICML 2022 | ⛔ deferred | multi-objective trust regions; not in BoTorch core — non-trivial port |

## Mixed / categorical (single-objective)
| Algorithm | Venue | Status | Notes |
|---|---|---|---|
| Single-task GP (mixed) | — | ✅ implemented | `single_obj_mixed_variable/single_task_gp` (optimize_acqf_mixed + snapping) |
| Random search (mixed) | — | ✅ implemented | |
| Probabilistic reparameterization | Daulton et al., NeurIPS 2022 | 🔜 follow-up | gradient-based discrete acqf opt; reimplement against BoTorch |
| Casmopolitan / Bounce | Wan 2021 / Papenmeier 2023 | ❌ excluded | dropped by request |

## Transform-based variants
Constrained, multi-objective, and mixed problems can also be run in a different category
via the wrappers in `bocode/transforms.py` (`PenalizedProblem`, `ScalarizedProblem`,
`ContinuousRelaxation`) — e.g. a constrained problem run with an unconstrained algorithm
through `PenalizedProblem`. See `Experiment_problems.md` for the full problem × variant
matrix.
