# import libraries
import os
import time
import numpy as np
from Bio import SeqIO

def letter_to_int(aln, alphabet='-ACDEFGHIKLMNPQRSTVWY'):
    if isinstance(alphabet, str):
        alphabet = list(alphabet)
    else:
        raise ValueError("'alphabet' must be a string.")

    letter_to_int = {letter: i for i, letter in enumerate(alphabet)}

    return np.vectorize(letter_to_int.get)(aln)

def compute_weights(aln, dist_threshold=0.3):
    M, N = aln.shape
    counts = np.zeros(M, dtype=np.int32)

    for i in range(M):
        dist = np.mean(aln != aln[i], axis=1)
        counts[i] = np.sum(dist < dist_threshold)

    counts[counts==0] = 1
    w = 1. / counts
    return w, w.sum()

def unique_indices_groups(x):
    unique, indices, inverse = np.unique(x, axis=0, return_index=True, return_inverse=True)
    groups = [[] for _ in range(inverse.max()+1)]
    for idx, group_id in enumerate(inverse):
        groups[group_id].append(idx)
    return unique, indices, groups

# import alignment from FASTA file
sequences = []
descriptions = []
aln_file = "./data/iter_aln_dedup_sp.faa"
print("Importing FASTA file...")
start = time.perf_counter()
for record in SeqIO.parse(aln_file, "fasta"):
    sequences.append([l for l in str(record.seq)])
    descriptions.append(record.description)
sequences = np.array(sequences)
descriptions = np.array(descriptions)
aln = letter_to_int(sequences)
desc = descriptions.copy()
M, N = aln.shape
stop = time.perf_counter()

print(f"It took {stop-start:.3f} seconds.")
print(f"FASTA file imported. There are {M} rows and {N} columns in the full alignment.")

# deduplicate the alignment
# full aln should already be deduplicated
#print("Deduplicating the alignment...")
#aln, aln_indices, aln_groups = unique_indices_groups(aln)
#desc = []
#for group in aln_groups:
#    desc.append(';'.join(descriptions[group]))
#desc = np.array(desc)
#M, N = aln.shape
#print(f"Alignment reduced to red sector and deduplicated. There are {M} rows and {N} columns in the reduced alignment.")

# compute weights
#start = time.perf_counter()
#w, M_eff = compute_weights(aln, dist_threshold=0.3)
#M_eff = int(M_eff)
#stop = time.perf_counter()
#print(f"It took {stop-start:.3f} seconds.")
#print(f"Computed weights. The effective number of sequences is {M_eff}.")

# for the full alignment we cannot compute the weights, they are found in a file
w = np.load("./data/full_weights.npy").ravel()
M_eff = int(np.sum(w))
print(f"Meff is {M_eff}")

# set the seed
np.random.seed(42)

# choose K subsets of size N_eff
print("Making subalignments...")
K = 10
p = w / np.sum(w)
subset_folder_name = "full_subalns"
path = "./data/"+subset_folder_name
if not os.path.exists(path):
    os.mkdir(path)
for idx in range(K):
    print(f"Preparing subalignment #{idx}...")
    subset_idx = np.random.choice(np.arange(M), size=M_eff, p=p)
    aln_subset = aln[subset_idx]
    desc_subset = desc[subset_idx]
    w_subset, M_eff_subset = compute_weights(aln_subset, dist_threshold=0.3)
    print(f"Subalignment has Meff={M_eff_subset}")

    np.save(f"./data/{subset_folder_name}/subaln{idx}_seq.npy", aln_subset)
    np.save(f"./data/{subset_folder_name}/subaln{idx}_desc.npy", desc_subset)
    np.save(f"./data/{subset_folder_name}/subaln{idx}_weights.npy", w_subset)

print(f"Subalignments and weights saved in folder {path}")
print("Done with the preparation of datasets!")
