#!/bin/bash
#SBATCH --job-name=plot_shuffle_full
#SBATCH --output=logs/plot_shuffle_full_%a.out
#SBATCH --error=logs/plot_shuffle_full_%a.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=10,20,30,40,235,245,255,260

#Pp72
#20,40,60,80,100,120,140,160,180,200,220,240

STEP=$((SLURM_ARRAY_TASK_ID))

python3 06a_plot_shuffling_full.py ${STEP} ./models/ ./figures/
