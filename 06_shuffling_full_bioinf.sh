#!/bin/bash
#SBATCH --job-name=shuffle_full
#SBATCH --output=logs/shuffle_full_%a.out
#SBATCH --error=logs/shuffle_full_%a.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=2200-2209,2400-2409

#Pp72
#20,40,60,80,100,120,140,160,180,200,220,240

SUBALN_IDX=$((SLURM_ARRAY_TASK_ID % 10))
STEP=$((SLURM_ARRAY_TASK_ID / 10))

echo "${STEP} ${SUBALN_IDX}"

python3 06_shuffling_full.py ${SUBALN_IDX} ${STEP} ./data/full_subalns

echo "SubAln ${SUBALN_IDX}, Step ${STEP} completed."
