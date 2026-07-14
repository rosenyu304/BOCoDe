# Reproducing the BoCoDe environment

Every result must be traceable to (a) a **commit SHA** and (b) **this environment**.
If either differs, the numbers are not comparable.

## The stack (validated on all six machines, 2026-07-13)

| package | version | why pinned |
|---|---|---|
| torch | **2.11.0+cu128** | 2.13 (plain PyPI) ships a `libtorch_cuda.so` whose NCCL symbols don't match the wheel it pulls (`undefined symbol: ncclCommResume`). The cu130 wheels need driver >= 580; our machines run 535-595. cu128 covers sm_86 (3090 Ti), sm_89 (4090), sm_90 (H100) and sm_120 (RTX 5090 Blackwell) — cu126 would NOT support the 5090. |
| botorch | 0.18.1 | dimension-scaled priors (needs >= 0.12) |
| gpytorch | 1.15.2 | matches botorch |
| pymoo | 0.6.2 | exact hypervolume (compiled WFG) — 10-220x faster than BoTorch's box decomposition, identical to ~1e-16 |
| xgboost | 3.3.0 | HPO-B surrogates (`HPOBSurr_*`) crash without it |

## Install

```bash
conda create -n bocode python=3.12 -y && conda activate bocode
pip install torch --index-url https://download.pytorch.org/whl/cu128   # MUST use this index
pip install -e .
pip install -r requirements-lock.txt      # exact pins
python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible'; print(torch.__version__)"
```

## The TabPFN (TFM) methods need a SEPARATE env

TabPFN and the GP stack cannot share one env. TFM methods (`git_bo`, `pfn_cei`,
`tfm_turbo`, `tfm_scbo`, `tfm_qnehvi`, `tfm_qnparego`, `tfm_cqnehvi`, `tfm_cqnparego`)
run in `r2_mixedbo` (python 3.11, `tabpfn 8.0.6` = TabPFN v3). They need a separate
submission lane on the cluster.

Checkpoint: `BOCODE_TABPFN_CKPT=/home/rosenyu/Downloads/tabpfn-v3-regressor-v3_default.ckpt`

> `r2_mixedbo` still has torch 2.6, on which `tfm_cqnehvi` (and stock botorch
> `constrained_qnehvi`) crash with a CUDA `index_put` assert. Upgrade that env to
> torch 2.11+cu128 as well, or run those `--device cpu`.

## Verifying a machine before it runs anything

```bash
./2026_07_Experiment/sync_machines.sh <sha>   # prints ✅/❌ per host by comparing git rev-parse HEAD
```
A run whose SHA != the campaign SHA is suspect and must be requeued. This exists because
an ad-hoc rsync silently left four machines on stale, sign-inverted code for hours.
