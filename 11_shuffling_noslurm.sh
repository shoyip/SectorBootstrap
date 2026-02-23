#!/bin/bash

# Sector subalignment shuffling SBM training - Local (no SLURM)
#
# 10 subalignments × 24 steps = 240 runs
# (23 columns marion_red_sector + 1 baseline)
#
# Steps:
#   0: No shuffling (baseline)
#   1-23: Progressive sector column shuffling (least relevant first)
#
# Parameters: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15
#
# Subalignments from ./data/subalns/ (created by 03_create_subalns.py)
# Already: sector-only, deduplicated, weighted sampled

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through all subalignments and steps
for SUBALN_IDX in {0..4}; do
    for STEP in {0,5,10,15,20,23}; do
        echo "========================================"
        echo "Subalignment: ${SUBALN_IDX}, Step: ${STEP}"
        echo "========================================"
        
        # Run the single-step training script
        python 11_shuffling_single.py ${SUBALN_IDX} ${STEP} --subaln_dir ./data/subalns \
            2>&1 | tee logs/sector_shuffle_${SUBALN_IDX}_${STEP}.log
        
        echo "SubAln ${SUBALN_IDX}, Step ${STEP} completed."
        echo ""
    done
done

echo "All training runs completed!"
