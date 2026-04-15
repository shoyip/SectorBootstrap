#!/bin/bash
#SBATCH --job-name=mutations_full
#SBATCH --output=logs/mutations_full_%a.out
#SBATCH --error=logs/mutations_full_%a.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0,50,100,150,200,210,220,230,240,250

STEP=$((SLURM_ARRAY_TASK_ID))

python3 07_mutations_full.py ${STEP}

echo "Step ${STEP} completed."
