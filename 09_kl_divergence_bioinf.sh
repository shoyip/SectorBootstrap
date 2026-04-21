#!/bin/bash
#SBATCH --job-name=kl_full
#SBATCH --output=logs/kl_full_%a.out
#SBATCH --error=logs/kl_full_%a.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-2609

SUBALN_IDX=$((SLURM_ARRAY_TASK_ID % 10))
STEP=$((SLURM_ARRAY_TASK_ID / 10))

echo "${STEP} ${SUBALN_IDX}"

python3 09_kl_divergence.py \
  "${SUBALN_IDX}" \
  "${STEP}" \
  --models-dir ./models \
  --train-dir ./data/FullSubAln \
  --generated-key Test \
  --output-dir ./results/kl_divergence

echo "KL done for SubAln ${SUBALN_IDX}, Step ${STEP}."
