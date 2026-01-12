# this code was partially generated using LLMs

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def _compute_weight_for_sequence(idx: int, data_encoded: np.ndarray, L: int, th: float) -> float:
    """Compute weight for a single sequence (worker function for multiprocessing)."""
    s = data_encoded[idx]  # Get i-th row (sequence)
    # Compare i-th sequence with all j-th sequences element-wise
    matches = (s == data_encoded)  # Broadcasting: (L,) == (N, L) -> (N, L) boolean array
    seq_id = np.sum(matches, axis=1) / L  # Sum matches per sequence and normalize
    n_clust = np.sum(seq_id > th)
    return 1.0 / n_clust


def compute_weights(
    data: np.ndarray,
    th: float = 0.8,
    n_processes: int = None,
) -> np.ndarray:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, 
    where 'n_clust' is the number of sequences that have a sequence identity with 's' >= th.
    
    Args:
        data (np.ndarray): Input dataset of shape (M, N) where M is the number of sequences
                          and N is the sequence length. Each element is a single character 
                          string representing an amino acid.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        n_processes (int, optional): Number of processes to use. Defaults to cpu_count().
    
    Returns:
        np.ndarray: Array with the weights of the sequences, shape (M, 1).
    """
    if len(data.shape) != 2:
        raise ValueError("'data' must be a 2D array of shape (M, N) where M is the number of sequences.")
    
    M, N = data.shape  # M sequences, each of length N
    
    # Use all available cores if not specified
    if n_processes is None:
        n_processes = cpu_count()
    
    # Create worker function with fixed arguments
    worker_func = partial(_compute_weight_for_sequence, 
                         data_encoded=data, 
                         L=N, 
                         th=th)
    
    # Parallel processing
    with Pool(processes=n_processes) as pool:
        #weights = pool.map(worker_func, range(M))
        weights = pool.map(worker_func, range(M), chunksize=max(1, M // (n_processes * 4)))

    
    # Return as column vector
    return np.array(weights, dtype=np.float32).reshape(-1, 1)

if __name__ == "__main__":
    aln = np.load("./data/full_aln.npz")
    start = time.perf_counter()
    w = compute_weights(aln['seq'][:5_000])
    end = time.perf_counter()
    print(end - start)
    print(np.sum(w))
    np.save('data/first_5000_weights.npy', w)
