#!/bin/bash
#SBATCH --job-name=sector_shuffle
#SBATCH --output=logs/sector_shuffle_%a.out
#SBATCH --error=logs/sector_shuffle_%a.err
#SBATCH --array=0-239
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate subalignment index and step from array task ID
SUBALN_IDX=$((SLURM_ARRAY_TASK_ID / 24))
STEP=$((SLURM_ARRAY_TASK_ID % 24))

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subalignment: ${SUBALN_IDX}"
echo "Step: ${STEP}"

# Run the single-step training script
python 05_shuffling.py ${SUBALN_IDX} ${STEP} --subaln_dir ./data/subalns

echo "SubAln ${SUBALN_IDX}, Step ${STEP} completed."
