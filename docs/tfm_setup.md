# Running the TFM (transformer-foundation-model) BO algorithms

`algorithms/single_obj/git_bo.py` (GIT-BO) and
`algorithms/single_obj_constrained/pfn_cei.py` (PFN-CEI) use a pretrained tabular
foundation model (**TabPFN**) as the surrogate instead of a Gaussian process. They
depend on a **TabPFN fork** that adds bar-distribution acquisition methods
(`ei`/`ucb`/`mean`/`variance`) — the stock PriorLabs TabPFN does not expose these.

## Why a separate environment

That fork pins **scikit-learn** to an older API (`sklearn.utils.validation._is_pandas_df`)
and **Python <3.12**. BoCoDe's core env is Python 3.12 with scikit-learn ≥1.9, where
the fork fails to import. So the TFM algorithms cannot run in the core `bocode` env —
use a dedicated environment.

## Setup (one-time)

```bash
mamba create -n bocode_tfm python=3.11 -y
mamba run -n bocode_tfm pip install -e .            # bocode itself (from the repo root)
# the TabPFN fork with bar-distribution acquisitions (path to the GIT-BO fork):
mamba run -n bocode_tfm pip install -e /path/to/GITBO/tabpfn
mamba run -n bocode_tfm pip install "scikit-learn<1.6"   # compatible with the fork's compat shim
```

TabPFN downloads its model weights from Hugging Face on first use. Inference runs on
GPU if available, otherwise CPU (slower — keep `--iters` modest for CPU smoke runs).

## Running

```bash
# GIT-BO, fixed rank 5 (original) and the Marzouk certified auto-rank:
mamba run -n bocode_tfm python -m algorithms.single_obj.git_bo --problem Ackley --iters 50 --rank 5
mamba run -n bocode_tfm python -m algorithms.single_obj.git_bo --problem Ackley --iters 50 --rank marzouk

# PFN-CEI on a constrained problem:
mamba run -n bocode_tfm python -m algorithms.single_obj_constrained.pfn_cei --problem PressureVessel --iters 50
```

All the common flags apply (`--seed`, `--show_progress`, `--saved_full_experiment`),
so the TFM runs log the same per-iteration `.npz` trace as the GP algorithms.

## What the two rank modes do (GIT-BO)

Both build the gradient-information matrix `H = (1/n) Σ (∂μ/∂x)(∂μ/∂x)ᵀ` from the
TabPFN posterior-mean gradient and sample candidates inside its top-`r` eigen-subspace.

- `--rank 5` — fixed `r = 5` (the original GIT-BO default).
- `--rank marzouk` — the **certified rank** `r* = min{r : Σ_{i>r} λ̄_i ≤ 2ε/κ}` on the
  trace-normalised spectrum (Zahm, Cui, Law, Spantini & Marzouk, *Math. Comp.* 2022),
  which keeps ≥ (1 − 2ε/κ) of the gradient-information energy with a KL bound — an
  error-controlled replacement for the fixed heuristic rank.
