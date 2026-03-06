#!/bin/bash
#SBATCH --job-name=subalns
#SBATCH --output=logs/subalns.out
#SBATCH --error=logs/subalns.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

python 02_subalns.py
