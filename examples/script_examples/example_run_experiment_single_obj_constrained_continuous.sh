#!/usr/bin/env bash
# Run the BoCoDe baselines on every single obj constrained continuous problem.
# Category filter: bocode.list_problems(num_objectives=1, constrained=True, input_type='continuous')
# See examples/README.md for the runner options.
set -e
cd "$(dirname "$0")/../.."   # -> repo root (BOCoDe/)

# ============================ LOCAL (CPU-capped) ============================
# Sequential runner; cap BLAS threads so it does not saturate the machine.
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
  python examples/run_experiments.py \
    --problems all --objectives 1 --constrained --input-type continuous \
    --seeds 0 1 2 --n-init 20 --iters 50 \
    --outdir examples/results

# ============================ SLURM (uncomment) ============================
# Save as a .slurm file (or add these #SBATCH lines at the very top) and `sbatch` it.
# #SBATCH --job-name=bocode_single_obj_constrained_continuous
# #SBATCH --array=0-15                 # 16 shards
# #SBATCH --cpus-per-task=12
# #SBATCH --mem=8G
# #SBATCH --time=12:00:00
# #SBATCH --output=logs/single_obj_constrained_continuous_%a.out
#
# source ~/miniforge3/etc/profile.d/conda.sh && mamba activate bocode
# OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
#   python examples/run_experiments.py \
#     --problems all --objectives 1 --constrained --input-type continuous \
#     --seeds 0 1 2 --n-init 20 --iters 50 \
#     --outdir examples/results \
#     --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"
