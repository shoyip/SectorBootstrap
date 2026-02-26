#!/bin/bash
#SBATCH --job-name=param_nchains
#SBATCH --output=logs/params_nchains_%a.out
#SBATCH --error=logs/params_nchains_%a.err
#SBATCH --array=10,100,500,1000,2000,5000
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

# Create logs directory if it doesn't exist
mkdir -p logs

# Calculate subalignment index and step from array task ID
SUBALN_IDX=0
N_CHAINS=$SLURM_ARRAY_TASK_ID

echo "N_chains: ${N_CHAINS}"
echo "Subalignment: ${SUBALN_IDX}"

# Activate your conda/virtual environment if needed
# source activate your_env
# or
# source /path/to/venv/bin/activate

# Run the single-step training script
python 03_paramsearch.py ${SUBALN_IDX} ${N_CHAINS}

echo "SubAln ${SUBALN_IDX}, N_chains ${N_CHAINS} completed."
