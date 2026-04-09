from pathlib import Path
import argparse

import numpy as np

from utils_SBM import run_SBM

current_dir = Path(__file__).resolve()
models_dir = current_dir / "models"

def shuffle_columns(aln, cols_to_shuffle):
    shuffled_aln = np.copy(aln)
    for col in cols_to_shuffle:
        np.random.shuffle(shuffled_aln[:, col])
    return shuffled_aln

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SBMs with full alignment with shuffled columns. Columns are shuffled in decreasing order with the component on the red sector.")
    parser.add_argument("subaln_idx", type=int, help="Index of previously generated subalignment (0-9)")
    parser.add_argument("step", type=int, help="Step of shuffling (0-260)")
    parser.add_argument("subaln_dir", type=str, default="./data/full_subalns", help="Directory where subalignments can be found.")
    args = parser.parse_args()

    step = args.step
    subaln_idx = args.subaln_idx

    subaln_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_seq.npy"
    aln = np.load(subaln_file, allow_pickle=True)
    M, L = aln.shape
    print(f"Loaded subalignment {args.subaln_idx}: {M} sequences, {L} positions")

    weights_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_weights.npy"
    weights = np.load(weights_file, allow_pickle=True)
    print(f"Loaded weights for subalignment {args.subaln_idx}, Meff={np.sum(weights)}")

    red_sector_eigenvalues = np.load("./data/red_sector_eigvals.npy")
    red_residues_sorted = np.argsort(red_sector_eigenvalues)
    cols_to_shuffle = red_residues_sorted[:step]
    
    print(f"Number of residues to shuffle: {step}")

    np.random.seed(42)

    print(f"Subaln #{subaln_idx}, step {step}")
    aln_shuffled = shuffle_columns(aln, cols_to_shuffle)
    run_SBM(
            aln_shuffled,
            fam = f"FullShuffling_SubAln{subaln_idx}_Step{step:02d}",
            weights = weights,
            N_chains = 50,
            k_MCMC = 100000,
            m = 1,
            N_iter = 400,
            Nb_av = 10,
            lambdJ = 0,
            lambdh = 0
    )
