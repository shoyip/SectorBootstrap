#!/bin/bash
#SBATCH --job-name=shuffling
#SBATCH --output=logs/shuffling_%a.out
#SBATCH --error=logs/shuffling_%a.err
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate subalignment index and step from array task ID
SUBALN_IDX=$((SLURM_ARRAY_TASK_ID / 23))
STEP=$((SLURM_ARRAY_TASK_ID % 23))

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Subalignment: ${SUBALN_IDX}"
echo "Step: ${STEP}"

# Run the single-step training script
python 05_shuffling.py ${SUBALN_IDX} ${STEP} ./data/subalns

echo "SubAln ${SUBALN_IDX}, Step ${STEP} completed."
