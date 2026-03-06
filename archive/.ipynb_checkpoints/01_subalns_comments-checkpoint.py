

# # import NumPy arrays from .npz file
# aln_file = input("Enter alignment .npz filename [./data/full_aln.npz]: ")
# if aln_file == "":
#     aln_file = "./data/full_aln.npz"
# aln = np.load(aln_file)


# # import weights from NumPy file
# w_file = input("Enter .npy file of weights: ")
# if w_file == "":
#     w_file = "./data/full_weights.npy"
# w = np.load(w_file)

#subset_indices = np.random.choice(np.arange(0, M), size=[K, M_eff], p=p)
#
## mask the description and sequence arrays to produce new sub-alignments
#aln_subsets = np.take(aln_redsector, subset_indices, axis=0)
#desc_subsets = np.take(desc_redsector, subset_indices, axis=0)
#
#print("Saving subalignments...")
#subset_folder_name = "subalns"
#path = "./data/"+subset_folder_name
#if not os.path.exists(path):
#    os.mkdir(path)
#
#for idx, (desc_subset, aln_subset) in enumerate(zip(desc_subsets, aln_subsets)):
#    np.save("./data/"+subset_folder_name+"/subaln"+str(idx)+"_seq.npy", aln_subset)
#    np.save("./data/"+subset_folder_name+"/subaln"+str(idx)+"_desc.npy", desc_subset)



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