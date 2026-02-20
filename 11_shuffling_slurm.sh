#!/bin/bash
#SBATCH --job-name=sector_shuffle
#SBATCH --output=logs/sector_shuffle_%a.out
#SBATCH --error=logs/sector_shuffle_%a.err
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Sector-only shuffling SBM training - SLURM array job
# Steps:
#   0: No shuffling (baseline)
#   1-23: Progressive sector column shuffling (least relevant first)
#
# Parameters: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15
#
# Preprocessing:
#   - Extract sector columns only (23 columns)
#   - Deduplicate identical sequences

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your conda/virtual environment if needed
# source activate your_env
# or
# source /path/to/venv/bin/activate

# Run the single-step training script
python 11_shuffling_single.py ${SLURM_ARRAY_TASK_ID} --aln_file ./data/full_aln.npz

echo "Step ${SLURM_ARRAY_TASK_ID} completed."
