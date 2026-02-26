# this is the code for the parameter search
# basically I will keep the parameters of the SBM to
# kMCMC=5000
# m=1
# N_iter=1000
# Nb_av=10
# lambdJ=0, lambdh=0
# theta not needed because I have precomputed weights
# and then change N_chains

import argparse
import numpy as np
from utils_SBM import run_SBM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training SBM with fixed parameters and multiple N_chains")
    parser.add_argument("subaln_idx", type=int, help="Subalignment index (0-9)")
    parser.add_argument("n_chains", type=int, help="Number of MCMC chains")
    args = parser.parse_args()
    subaln_file = f"./data/subalns/subaln{args.subaln_idx}_seq.npy"
    aln = np.load(subaln_file, allow_pickle=True)
    M, N_sector = aln.shape

    w_file = f"./data/subalns/subaln{args.subaln_idx}_weights.npy"
    w = np.load(w_file, allow_pickle=True)

    run_SBM(aln, fam="SectorSubaln0", weights=w, N_chains=args.n_chains)
