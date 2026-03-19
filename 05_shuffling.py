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
    parser = argparse.ArgumentParser(description="Train SBMs with alignments that have shuffled columns.")
    parser.add_argument("subaln_idx", type=int, help="Index of previously generated subalignment (0-9)")
    parser.add_argument("step", type=int, help="Step number from least to most important residue (0-22)")
    parser.add_argument("subaln_dir", type=str, default="./data/subalns", help="Directory where subalignments can be found.")
    args = parser.parse_args()

    subaln_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_seq.npy"
    aln = np.load(subaln_file, allow_pickle=True)
    M, L = aln.shape
    print(f"Loaded subalignment {args.subaln_idx}: {M} sequences, {L} positions")

    weights_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_weights.npy"
    weights = np.load(weights_file, allow_pickle=True)
    print(f"Loaded weights for subalignment {args.subaln_idx}, Meff={np.sum(weights)}")

    sector_by_relevance = np.load("./data/red_sector_indices.npy")[::-1]
    rank_sector = np.empty_like(sector_by_relevance)
    rank_sector[np.argsort(sector_by_relevance)] = np.arange(len(sector_by_relevance))
    print(f"Shuffling order: {sector_by_relevance}")
    print(f"Number of residues to shuffle: {len(sector_by_relevance)}")

    np.random.seed(42)

    step = args.step
    subaln_idx = args.subaln_idx

    if step == 0:
        print(f"subaln #{subaln_idx}, step {step}, no shuffling...")
        run_SBM(
                aln,
                fam = f"SectorShuffling_SubAln{subaln_idx}_Step{step:02d}_noshuffle",
                weights = weights,
                N_chains = 75,
                k_MCMC = 5000,
                m = 1,
                N_iter = 1000,
                Nb_av = 1,
                lambdJ = 0,
                lambdh = 0
        )
    else:
        sector_idx = step - 1
        print(f"subaln #{subaln_idx}, step {step}, column {sector_by_relevance[sector_idx]} shuffled...")
        cols_to_shuffle = rank_sector[:sector_idx+1]
        col = sector_by_relevance[sector_idx]
        aln_shuffled = shuffle_columns(aln, cols_to_shuffle)
        run_SBM(
                aln_shuffled,
                fam = f"SectorShuffling_SubAln{subaln_idx}_Step{step:02d}_Col{col}",
                weights = weights,
                N_chains = 75,
                k_MCMC = 5000,
                m = 1,
                N_iter = 1000,
                Nb_av = 10,
                lambdJ = 0,
                lambdh = 0
        )
