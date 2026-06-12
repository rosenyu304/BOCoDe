# BOCoDe: Benchmarks for Optimization and Computational Design

[![Python](https://img.shields.io/pypi/pyversions/bocode.svg)](https://badge.fury.io/py/bocode)
![tests](https://github.com/rosenyu304/bocode/workflows/Python%20Tests/badge.svg)
[![code style: Ruff](
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](
    https://github.com/astral-sh/ruff)

BOCoDe is a Python/PyTorch library of **real-world** optimization problems for
benchmarking optimization algorithms, with a first-class BoTorch/Ax interface for
Bayesian optimization research. It collects engineering design, control,
materials-science, hyperparameter, and combinatorial problems drawn from the
literature — deliberately excluding purely synthetic test functions (Ackley,
Rosenbrock, ZDT, DTLZ, …) so that benchmarks reflect problems people actually
solve.

> [!IMPORTANT]
> Output objective values are meant to be **maximized** (the library was designed
> for Bayesian optimization). For minimization algorithms, negate the objective.
> Constraints are **inequality constraints**: `gx <= 0` is feasible.

> [!NOTE]
> This branch (`dev/2026_06`) is a substantial restructuring. See
> [What changed](#-what-changed-dev2026_06) below.

# 💡 What is in BOCoDe?

Every problem is a real-world problem with a cited source. Problems are grouped by
application area and accessed by name through a central registry:

| Area | Examples | Count |
|---|---|---|
| **Engineering** | trusses, pressure vessel, speed reducer, Mazda & MOPTA08 car design, CEC2020 real-world constrained suite (57), RE/CRE multi-objective suite (24), welded beam, disc brake | ~133 |
| **Control** | MuJoCo locomotion (Ant, HalfCheetah, Hopper, …), CartPole / Acrobot PID tuning | ~18 |
| **Materials** | AgNP, CrossedBarrel, P3HT, Perovskite, AutoAM (PV-Lab experimental datasets) | 5 |
| **Hyperparameter Optimization** | SVM, LassoBench real datasets (DNA, Leukemia, RCV1, …) | 6 |
| **Combinatorial** | Traveling Salesman (51 / 100 cities) | 2 |

See **[CATEGORIZATION.md](CATEGORIZATION.md)** for the full per-problem table
(dimensions, objectives, constraints, variable types, convexity, NP-hardness)
and the NP-hardness assessment methodology.

# 💻 Installation

Core install (enough to import `bocode` and run most problems):

```bash
pip install bocode
```

Some problems need heavy or native dependencies, shipped as optional **extras**.
Install only what you need:

```bash
pip install "bocode[mujoco]"     # MuJoCo control problems
pip install "bocode[truss]"      # Truss10D..Truss200D
pip install "bocode[box2d]"      # RobotPush
pip install "bocode[modact]"     # MODAct suite
pip install "bocode[hpo]"        # SVM
pip install "bocode[mazda]"      # Mazda car design
pip install "bocode[neorl]"      # QPowerModel surrogate
pip install "bocode[all]"        # everything available on PyPI
```

| Extra | Enables | Backing dependency |
|---|---|---|
| `mujoco` | MuJoCo locomotion problems | `gymnasium[mujoco]` |
| `control` | CartPole / Acrobot | `gymnasium` |
| `truss` | Truss10D…Truss200D | `slientruss3d` |
| `box2d` | RobotPush | `Box2D`, `pygame`, `joblib` |
| `modact` | MODAct CS/CT/CTS/CTSE/CTSEI | `modact` |
| `hpo` | SVM | `scikit-learn` |
| `mazda` | Mazda | `openpyxl` |
| `neorl` | QPowerModel | `onnxruntime` |
| `lasso` | LassoBench real datasets | `LassoBench` (git, not on PyPI) |
| `viz` | function visualization | `dash`, `plotly`, `matplotlib` |

`LassoBench` is not on PyPI, so the `lasso` extra installs it from git:

```bash
pip install "bocode[lasso]"
```

Accessing a problem without its extra installed raises a clear error telling you
which extra to install.

### Large data files

A few problems use large data (the SVM dataset, the MOPTA08 and Mazda binaries).
These are **not shipped** in the package; they are downloaded on first use to
`~/.cache/bocode` (override with `BOCODE_CACHE_DIR`) and verified by checksum. Set
`BOCODE_DATA_BASE_URL` to point at a mirror if needed.

# 🔍 Example Usage

Problems are accessed by name (flat API) or via the registry:

```python
import bocode
import torch

# List and filter problems by metadata
bocode.list_problems(application="Engineering")
bocode.list_problems(num_objectives=2, constrained=True)

# Instantiate by name (both forms are equivalent)
problem = bocode.Car()
problem = bocode.get_problem("Car")()

# Inspect metadata
meta = bocode.get_metadata("Car")     # dim, #obj, #constraints, bounds, source, ...

# Evaluate
x = torch.rand(5, problem.dim)
x = problem.scale(x)                   # map [0, 1] samples into the problem bounds
values, constraints = problem.evaluate(x)
print("feasible:", (constraints <= 0).all(dim=1))
```

Convenience selectors for algorithm benchmarking:

```python
bocode.get_single_objective_unconstrained()
bocode.get_single_objective_constrained()
bocode.get_multi_objective_unconstrained()
bocode.get_multi_objective_constrained()
```

Materials problems are discrete dataset-lookup problems — optimize over the
measured candidate set:

```python
p = bocode.AgNP()
values, _ = p.evaluate(p.candidates[:10])   # measured objective of those candidates
```

> The old `bocode.Engineering.Car` namespace still works but emits a
> `DeprecationWarning`; prefer `bocode.Car` / `bocode.get_problem("Car")`.

# 🧮 Algorithms

`algorithms/` holds single-file, CleanRL-style BO baselines (random search, TuRBO,
Vanilla BO, SCBO, qEHVI/qNEHVI, …) organized by problem category. This is
research code, separate from the installable package. See
[algorithms/README.md](algorithms/README.md). *(Scaffolding on this branch;
implementations land in Push 2 — see the roadmap.)*

# 🗺️ Roadmap

- **Foundation (this push)** — restructure into `bocode/opt_problems/`, remove
  synthetic problems, per-problem JSON metadata + registry, CATEGORIZATION.md,
  slim packaging with extras, download-on-demand data, materials dataset problems.
- **Push 2 — GP algorithms.** CleanRL-style baselines under `algorithms/`, each
  with problem-optimization and dataset-optimization variants.
- **Push 3 — more problems & methods.** LassoBench modernization, minimdo
  engineering problems, firefly-paper problems (mixed-integer + continuous), and
  transformer-based methods (GIT-BO, PFN-CEI / TabPFN).

# 🛠️ Development

```bash
mamba create -n bocode python=3.12 -y
mamba run -n bocode pip install -e ".[all]" pytest pytest-cov ruff
mamba run -n bocode pytest tests/        # smoke test skips problems whose extra is absent
```

Regenerate the registry, metadata, and categorization table after adding a
problem:

```bash
python tools/generate_registry.py
python tools/generate_metadata.py
python tools/render_categorization.py
```

We welcome contributions! New problems should be real-world problems with a cited
source, one file per problem, following the existing `"""Sources: ..."""`
docstring convention.

# Citing

```bibtex
@misc{yu2025bocode,
    author={Rosen Ting-Ying Yu, Advaith Narayanan, Cyril Picard, Faez Ahmed},
    title = {{BOCoDe}: Benchmarks for Optimization and Computational Design},
    year={2025},
    url={https://github.com/rosenyu304/BOCoDe}
}
```

If you use a BOCoDe problem derived from another library or paper (BoTorch,
MODAct, LassoBench, the PV-Lab materials datasets, the CEC2020 / RE suites, …),
please also cite that source — each problem records its citation in its docstring
and metadata.

# License

BOCoDe is MIT licensed, as found in [LICENSE](LICENSE).
