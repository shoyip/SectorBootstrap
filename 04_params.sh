#!/bin/bash
#SBATCH --job-name=params
#SBATCH --output=logs/params_%a.out
#SBATCH --error=logs/params_%a.err
#SBATCH --array=1,2,4,8,16,32,64,128,256,512,1024,2048,4096
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

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
python 04_params.py ${SUBALN_IDX} ${N_CHAINS}

echo "SubAln ${SUBALN_IDX}, N_chains ${N_CHAINS} completed."
