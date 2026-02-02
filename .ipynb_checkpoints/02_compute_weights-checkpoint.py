import time
import torch
import numpy as np
from typing import Union

def letters_to_int(aln, alphabet='-ACDEFGHIKLMNPQRSTVWY'):
    if isinstance(alphabet, str):
        alphabet = list(alphabet)
    else:
        raise ValueError("'alphabet' must be a string.")

    letter_to_int = {letter: i for i, letter in enumerate(alphabet)}

    return np.vectorize(letter_to_int.get)(aln)

@torch.jit.script
def _get_sequence_weights(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
    seq_id = torch.sum(s == data, dim=1) / L
    n_clust = torch.sum(seq_id > th)

    return 1.0 / n_clust

def compute_weights(
        data: Union[np.ndarray, torch.Tensor],
        th: float = 0.8,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32
    ):
    if len(data.shape) not in (2, 3):
        raise ValueError("'data' must be either a (batch_size, L) or a (batch_size, L, q) ohe array.")
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, device=device)
    if len(data.shape) == 3:
        data_encoded = data.argmax(dim=2)
    else:
        data_encoded = data
    _, L = data_encoded.shape
    weights = torch.vstack([_get_sequence_weights(s, data_encoded, L, th) for s in data_encoded])

    return weights.to(dtype)

if __name__ == "__main__":
    aln = np.load('./data/full_aln.npz')
    aln_seq_int = letters_to_int(aln['seq'])
    start = time.perf_counter()
    w_t = compute_weights(aln_seq_int)
    w = w_t.cpu().numpy()
    end = time.perf_counter()
    w_file = input("Filename of weights: ")
    np.save(w_file, w)
    print(f"It took {end-start} seconds to process {aln_seq_int.shape[0]} sequences.")
    print(f"Neff is {np.sum(w)}.")
