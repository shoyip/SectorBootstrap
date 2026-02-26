# import libraries
import numpy as np
from Bio import SeqIO

# take filename of source alignment
aln_file = input("Enter filename of alignment: ")

# set default parameters
N, M = 0, 0
desc_list = []
seq_list = []
prev_length = 0

# scan through the alignment
for record_index, record in enumerate(SeqIO.parse(aln_file, "fasta")):
    # check if sequences are all of the same length
    # stop at the first occurrence of a sequence of different length
    current_len = len(str(record.seq))
    if (current_len != prev_length) & (record_index > 0):
        raise ValueError(f"Sequence of entry {record_index} has different length ({current_len}) wrt previous entries.")

    # append to list
    desc_list.append(str(record.description))
    seq_list.append(str(record.seq))

    # keep track of sequence length
    prev_length = len(str(record.seq))

    M += 1
N = prev_length

print(f"There are {M} sequences and {N} positions.")

aln_array_file = input("Enter filename of alignment array (.npz): ")

# convert lists to NumPy arrays
aln_array_desc = np.array(desc_list)
aln_array_seq = np.array([[res for res in seq] for seq in seq_list])

# save NumPy arrays in a .npz file
np.savez_compressed(aln_array_file, desc=aln_array_desc, seq=aln_array_seq)
