#!/bin/bash
#SBATCH --job-name=sector
#SBATCH --output=logs/sector.out
#SBATCH --error=logs/sector.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

python 01_sector.py
