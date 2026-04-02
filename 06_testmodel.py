from pathlib import Path
import argparse

import numpy as np

from utils_SBM import run_SBM

current_dir = Path(__file__).resolve()
models_dir = current_dir / "models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SBMs with alignments that have shuffled columns.")
    parser.add_argument("subaln_idx", type=int, help="Index of previously generated subalignment (0-9)")
    parser.add_argument("subaln_dir", type=str, default="./data/subalns", help="Directory where subalignments can be found.")
    args = parser.parse_args()

    subaln_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_seq.npy"
    aln = np.load(subaln_file, allow_pickle=True)
    M, L = aln.shape

    weights_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_weights.npy"
    weights = np.load(weights_file, allow_pickle=True)
    print(f"Loaded weights for subalignment {args.subaln_idx}, Meff={np.sum(weights)}")

    np.random.seed(42)

    subaln_idx = args.subaln_idx

    print(f"Subaln 0 trained with one average")
    run_SBM(
            aln,
            fam = f"TestOfAln_20260401",
            weights = weights,
            N_chains = 256,
            k_MCMC = 10000,
            m = 1,
            N_iter = 1000,
            Nb_av = 1,
            lambdJ = 0,
            lambdh = 0
    )
