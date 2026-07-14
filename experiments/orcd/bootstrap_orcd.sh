#!/usr/bin/env bash
# Bootstrap BoCoDe on ORCD. Run ON the login node, inside tmux (it takes ~10 min).
#   tmux new -d -s boot '/orcd/data/faez/001/rosen/bocode/bootstrap_orcd.sh'
set -uo pipefail

ROOT=/orcd/data/faez/001/rosen/bocode
BRANCH=dev/2026_06
REPO=https://github.com/rosenyu304/BOCoDe.git
SHA=$(cat "$ROOT/CAMPAIGN_SHA")

module load miniforge/25.11.0-0

mkdir -p "$ROOT" "$ROOT/Results" "$ROOT/checkpoints" "$ROOT/logs"
cd "$ROOT"

# ---- code: pinned to the campaign SHA (never rsync — a SHA is verifiable) ----
if [ ! -d BOCoDe/.git ]; then
  git clone -q --branch "$BRANCH" "$REPO" BOCoDe
fi
cd BOCoDe
git fetch -q origin "$BRANCH"
git checkout -q --detach "$SHA"
git clean -qfd
echo "code at $(git rev-parse --short HEAD) (want ${SHA:0:8})"
cd "$ROOT"

# ---- env ----
# IMPORTANT: do NOT touch torch on ORCD. The cluster driver is 590, so cu130 works here and
# that is the stack the completed GPU campaign ran on. Only cap torch on the WORKSTATIONS.
if [ ! -d env ]; then
  conda create -y -q -p "$ROOT/env" python=3.12 >/dev/null
fi
source activate "$ROOT/env"
python -m pip -q install --upgrade pip
python -c "import torch" 2>/dev/null || python -m pip -q install torch
python -c "import botorch" 2>/dev/null || python -m pip -q install botorch pymoo xgboost
python -m pip -q install -e ./BOCoDe

python - <<'PY'
import torch, bocode, pymoo
print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"  bocode {len(bocode.PROBLEM_REGISTRY)} problems | pymoo {pymoo.__version__}")
PY
echo "BOOTSTRAP_DONE"
