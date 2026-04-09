#!/bin/bash
#SBATCH --job-name=subalns
#SBATCH --output=logs/subalns.out
#SBATCH --error=logs/subalns.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=Pdef

python3 02a_fullsubalns.py
