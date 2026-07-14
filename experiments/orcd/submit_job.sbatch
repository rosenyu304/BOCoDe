#!/bin/bash
#SBATCH --job-name=bocode
#SBATCH --requeue                       # MANDATORY on preemptable: come back after a kill
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# One (problem, algorithm, seed) run. Resumable: if this job is preempted and requeued,
# it picks up from its checkpoint instead of restarting. Verified: resume-not-restart.
#
# Usage (set by launch.py, not by hand):
#   sbatch --partition=... --gres=gpu:1 --cpus-per-task=4 --time=... \
#          --export=ALL,PROBLEM=RE24,ALGO=qnehvi,SEED=0,ITERS=1000,SHA=<sha>,DEVICE=cuda \
#          submit_job.sbatch

set -uo pipefail

ROOT=/orcd/data/faez/001/rosen/bocode
REPO=$ROOT/BOCoDe
RESULTS=$ROOT/Results
CKPT=$ROOT/checkpoints

module load miniforge/25.11.0-0 2>/dev/null
source activate $ROOT/env 2>/dev/null || conda activate $ROOT/env

# ---- provenance: a result is only valid if we know WHICH code produced it ----
ACTUAL_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ "$ACTUAL_SHA" != "$SHA" ]; then
  echo "FATAL: repo is at $ACTUAL_SHA but the campaign SHA is $SHA. Refusing to run."
  exit 90                                   # do NOT silently produce data from the wrong code
fi

echo "host=$(hostname) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} sha=$ACTUAL_SHA"
echo "problem=$PROBLEM algo=$ALGO seed=$SEED device=$DEVICE"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "cpu-only job"

mkdir -p "$RESULTS/$PROBLEM/$ALGO" "$CKPT/$PROBLEM/$ALGO"

# thread cap: 4 threads/job is the measured throughput optimum when packing a GPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1                   # without this the log looks frozen
export PYTHONPATH=$REPO

python "$ROOT/run_one.py" \
  --problem "$PROBLEM" --algo "$ALGO" --seed "$SEED" --iters "$ITERS" \
  --device "$DEVICE" \
  --results "$RESULTS" \
  --checkpoint "$CKPT/$PROBLEM/$ALGO/seed$SEED.ckpt" \
  --sha "$SHA"
rc=$?

# 0 = done. Anything else and the watchdog will see no .npz and resubmit.
echo "EXIT rc=$rc"
exit $rc
