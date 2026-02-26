#!/bin/bash
#SBATCH --job-name=sector_shuffle
#SBATCH --output=logs/sector_shuffle_%a.out
#SBATCH --error=logs/sector_shuffle_%a.err
#SBATCH --array=0-239
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Sector subalignment shuffling SBM training - SLURM array job
#
# 10 subalignments × 24 steps = 240 jobs (array 0-239)
# (23 columns marion_red_sector + 1 baseline)
#
# Array index mapping:
#   task_id = subaln_idx * 24 + step
#   subaln_idx = task_id // 24  (0-9)
#   step = task_id % 24         (0-23)
#
# Steps:
#   0: No shuffling (baseline)
#   1-23: Progressive sector column shuffling (least relevant first)
#
# Parameters: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15
#
# Subalignments from ./data/subalns/ (created by 03_create_subalns.py)
# Already: sector-only, deduplicated, weighted sampled

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate subalignment index and step from array task ID
SUBALN_IDX=$((SLURM_ARRAY_TASK_ID / 24))
STEP=$((SLURM_ARRAY_TASK_ID % 24))

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subalignment: ${SUBALN_IDX}"
echo "Step: ${STEP}"

# Activate your conda/virtual environment if needed
# source activate your_env
# or
# source /path/to/venv/bin/activate

# Run the single-step training script
python 11_shuffling_single.py ${SUBALN_IDX} ${STEP} --subaln_dir ./data/subalns

echo "SubAln ${SUBALN_IDX}, Step ${STEP} completed."
