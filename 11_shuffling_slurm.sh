#!/bin/bash
#SBATCH --job-name=shuffling_sbm
#SBATCH --output=logs/shuffling_%a.out
#SBATCH --error=logs/shuffling_%a.err
#SBATCH --array=0-25
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Shuffling SBM training - SLURM array job
# Steps:
#   0: No shuffling (param set A)
#   1: Rest shuffled (param set A)
#   2: Rest shuffled (param set B)
#   3-25: Progressive sector column shuffling (param set B)

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda/virtual environment if needed
# source activate your_env
# or
# source /path/to/venv/bin/activate

# Run the single-step training script
python 10_shuffling_single.py ${SLURM_ARRAY_TASK_ID} --aln_file ./data/full_aln.npz --M_eff 17163

echo "Step ${SLURM_ARRAY_TASK_ID} completed."
