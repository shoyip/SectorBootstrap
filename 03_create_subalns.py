# import libraries
import numpy as np

# import NumPy arrays from .npz file
aln_file = input("Enter alignment .npz filename: ")
aln = np.load(aln_file)

M, N = aln['seq'].shape

# enter the effective number of sequences
M_eff = int(input("Enter effective number of sequences (M_eff): "))

# enter the desired number of subsets
K = int(input("Enter desired number of subsets (K): "))

# set the seed
np.random.seed(42)

# choose K sets of size N_eff
subset_indices = np.random.randint(0, M, size=[K, M_eff])

# mask the description and sequence arrays to produce new sub-alignments
aln_seq_subsets = np.take(aln['seq'], subset_indices, axis=0)
aln_desc_subsets = np.take(aln['desc'], subset_indices, axis=0)

# save the subsets of the alignment in a .npz file
aln_subsets_file = input("Enter alignment subsets folder name: ")
for idx, aln_desc_subset, aln_seq_subset in enumerate(zip(aln_desc_subsets, aln_seq_subsets)):
    np.save("./data/"+aln_subsets_file+"/subaln_seq_"+str(idx)+(".npy"), aln_seq_subset)
    np.save("./data/"+aln_subsets_file+"/subaln_desc_"+str(idx)+(".npy"), aln_desc_subset)
