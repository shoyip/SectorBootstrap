#!/bin/bash
#SBATCH --job-name=create_subalns
#SBATCH --output=logs/subalns.out
#SBATCH --error=logs/subalns.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

python 03_create_subalns.py
