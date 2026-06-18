#!/usr/bin/env bash
# Run the BoCoDe baselines on every multi obj unconstrained continuous problem.
# Category filter: bocode.list_problems(num_objectives>=2 constrained=False input_type=continuous)
set -e
cd "$(dirname "$0")/../.."

# ============================ LOCAL (CPU-capped) ============================
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMEXPR_NUM_THREADS=12 \
  python examples/run_experiments.py \
    --problems all --objectives 2 --unconstrained --input-type continuous \
    --seeds 0 1 2 --n-init 20 --iters 50 \
    --outdir examples/results

# ============================ SLURM (uncomment) ============================
# #SBATCH --job-name=bocode_multi_obj_unconstrained_continuous
# #SBATCH --array=0-15
# #SBATCH --cpus-per-task=12
# #SBATCH --mem=8G
# #SBATCH --time=12:00:00
# #SBATCH --output=logs/multi_obj_unconstrained_continuous_%a.out
# source ~/miniforge3/etc/profile.d/conda.sh && mamba activate bocode
# OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
#   python examples/run_experiments.py \
#     --problems all --objectives 2 --unconstrained --input-type continuous --seeds 0 1 2 --n-init 20 --iters 50 \
#     --outdir examples/results \
#     --task-id "$SLURM_ARRAY_TASK_ID" --num-tasks "$SLURM_ARRAY_TASK_COUNT"
