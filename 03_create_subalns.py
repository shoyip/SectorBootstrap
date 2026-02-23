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

# # import NumPy arrays from .npz file
# aln_file = input("Enter alignment .npz filename [./data/full_aln.npz]: ")
# if aln_file == "":
#     aln_file = "./data/full_aln.npz"
# aln = np.load(aln_file)

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
M, N = aln.shape
stop = time.perf_counter()

print(f"It took {stop-start:.3f} seconds.")
print(f"FASTA file imported. There are {M} rows and {N} columns in the full alignment.")

# # import weights from NumPy file
# w_file = input("Enter .npy file of weights: ")
# if w_file == "":
#     w_file = "./data/full_weights.npy"
# w = np.load(w_file)

# define sector and reduce the alignment only to sector columns
print("Reducing the alignment to only columns of the red sector...")
redsector = [1, 2, 164, 165, 176, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 238, 239]
aln_redsector = aln[:, redsector]

# deduplicate the alignment
print("Deduplicating the alignment...")
aln_redsector, _, aln_groups = unique_indices_groups(aln_redsector)
desc_redsector = []
for group in aln_groups:
    desc_redsector.append(';'.join(descriptions[group]))
desc_redsector = np.array(desc_redsector)
M, N = aln_redsector.shape
print(f"Alignment reduced to red sector and deduplicated. There are {M} rows and {N} columns in the reduced alignment.")

# compute weights
start = time.perf_counter()
w, M_eff = compute_weights(aln_redsector, dist_threshold=0.15)
M_eff = int(M_eff)
stop = time.perf_counter()
print(f"It took {stop-start:.3f} seconds.")
print(f"Computed weights. The effective number of sequences is {M_eff}.")

# set the seed
np.random.seed(42)

# choose K subsets of size N_eff
print("Making subalignments...")
K = 10
p = w / np.sum(w)
subset_indices = np.random.choice(np.arange(0, M), size=[K, M_eff], p=p)

# mask the description and sequence arrays to produce new sub-alignments
aln_subsets = letter_to_int(np.take(aln_redsector, subset_indices, axis=0))
desc_subsets = np.take(desc_redsector, subset_indices, axis=0)

print("Saving subalignments...")
subset_folder_name = "subalns"
path = "./data/"+subset_folder_name
if not os.path.exists(path):
    os.mkdir(path)

for idx, (desc_subset, aln_subset) in enumerate(zip(desc_subsets, aln_subsets)):
    np.save("./data/"+subset_folder_name+"/subaln"+str(idx)+"_seq.npy", aln_subset)
    np.save("./data/"+subset_folder_name+"/subaln"+str(idx)+"_desc.npy", desc_subset)

print("Subalignments saved in folder ./data/subalns")
print("Done with the preparation of datasets!")

# for M_sub in [1_000, 2_000, 4_000, 6_000, 8_000]:
#     print(M_sub)
#     start = time.perf_counter()
#     weights, M_eff = compute_weights(aln[:M_sub, :])
#     stop = time.perf_counter()
#     print(int(stop-start))

# # enter the effective number of sequences
# M_eff = input("Enter effective number of sequences (M_eff) [17163]: ")
# if M_eff == "":
#     M_eff = 17163
# else:
#     M_eff = int(M_eff)
# 
# # enter the desired number of subsets
# K = input("Enter desired number of subsets (K) [10]: ")
# if K == "":
#     K = 10
# else:
#     K = int(K)
# 
# # enter the sector residues
# sector = input("Enter sector residues separated by commas [red_sector/all/list]: ")
# if sector == "":
# sector = [1, 2, 164, 165, 176, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 238, 239]
# elif sector == "all":
#     sector = list(np.arange(0, N))
# else:
#     sector = map(int, sector.split(","))
# sector = sorted(sector)
# 
# # set the seed
# np.random.seed(42)
# 
# # choose K sets of size N_eff
# subset_indices = np.random.randint(0, M, size=[K, M_eff])
# 
# # mask the description and sequence arrays to produce new sub-alignments
# aln_seq_subsets = letters_to_int(np.take(aln['seq'], subset_indices, axis=0))
# aln_desc_subsets = np.take(aln['desc'], subset_indices, axis=0)
# 
# # save the subsets of the alignment in a .npz file
# aln_subsets_file = input("Enter alignment subsets folder name: ")
# if aln_subsets_file == "":
#     aln_subsets_file = "iter_aln"
# if not os.path.exists("./data/"+aln_subsets_file):
#     os.mkdir("./data/"+aln_subsets_file)
# for idx, (aln_desc_subset, aln_seq_subset) in enumerate(zip(aln_desc_subsets, aln_seq_subsets)):
#     np.save("./data/"+aln_subsets_file+"/subaln_seq_"+str(idx)+(".npy"), aln_seq_subset[:, sector])
#     np.save("./data/"+aln_subsets_file+"/subaln_desc_"+str(idx)+(".npy"), aln_desc_subset)
