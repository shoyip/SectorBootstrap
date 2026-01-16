from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from Bio import SeqIO

AA_ALPHABET = '-ACDEFGHIKLMNPQRSTVWY'
aa_to_int = {v: k for k, v in dict(enumerate(AA_ALPHABET)).items()}

red_sector = [197, 239, 237, 224, 186, 225, 189, 190, 200, 227, 228, 222, 238, 2, 229, 164, 195, 194, 231, 165, 176, 1]
red_sector = sorted(red_sector)

def ohe_seq(seq, alphabet=AA_ALPHABET):
    q = len(alphabet)
    L = len(seq)
    int_seq = np.array([aa_to_int[res] for res in seq])
    ohe_seq = np.eye(q)[int_seq].ravel()
    return ohe_seq

def get_energy(seq, J, h):
    L, _, q, _ = J.shape
    J = J.transpose((0,2,1,3)).reshape((L*q, L*q))
    h = h.ravel()
    x = ohe_seq(seq)
    energy = - x@h - .5 * x@J@x
    return energy

def get_params_gauge(J, h):
    Jg, hg = np.copy(J), np.copy(h)
    
    h_mean_pos = np.mean(h, axis=1, keepdims=True)
    J_mean_axis2 = np.mean(J, axis=2, keepdims=True)
    J_mean_axis3 = np.mean(J, axis=3, keepdims=True)
    J_mean_all = np.mean(J, axis=(2,3))
    J_corr = np.sum(np.mean(J, axis=3) - np.expand_dims(J_mean_all, axis=2), axis=1)
    
    hg = hg - h_mean_pos + J_corr
    Jg = Jg - J_mean_axis2 - J_mean_axis3 + np.expand_dims(np.expand_dims(J_mean_all, axis=2), axis=3)

    return Jg, hg

def compute_all_single_mutations_deltaE(seq, h, J, aa_order):
    """
    Compute deltaE for all possible single mutations in the sequence.
    """
    L = len(seq)
    aa_to_idx = {aa:i for i, aa in enumerate(aa_order)}
    seq_idx = np.array([aa_to_idx[aa] for aa in seq])
    
    deltaEs = np.zeros((L, len(aa_order)))
    
    for i in range(L):
        for a, aa in enumerate(aa_order):
            if a == seq_idx[i]:
                deltaEs[i,a] = 0
                continue
            
            # Δh contribution
            dE = - (h[i,a] - h[i, seq_idx[i]])
            
            # ΔJ contributions
            for j in range(L):
                if j == i:
                    continue
                dE -= (J[i,j,a,seq_idx[j]] - J[i,j,seq_idx[i], seq_idx[j]])
            
            deltaEs[i,a] = dE
    
    return deltaEs

def get_best_indices(mat):
    """
    Return the (row, col) index of the smallest value in a 2D numpy array.
    """
    flat = mat.ravel()
    smallest_flat_idx = np.argmin(flat)
    smallest = divmod(smallest_flat_idx, mat.shape[1])
    return smallest

if __name__ == "__main__":
    rat_trypsin = "IVGGYTCQENSVPYQVSLNS-----GYHFCGGSLINDQWVVSAAHCYKS-------RIQVRLGEHNIN-VLEGNEQFVNAAKIIKHPNFDR--KTLNNDIMLIKLSSPVKLNARVATV-ALPS---SCAP-AG-TQCLISGWGNTLSSG----VNEPDLLQCLDAPLLPQADCEASYP--GKITDNMVCVGFLEGGKDSCQGDSGGPVVCN-----GELQGIVSWGY--GCALPDNPGVYTKVCNYVDWIQDTIAAN---"
    nums = ["16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "36A", "36B", "36C", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "61A", "61B", "61C", "61D", "61E", "61F", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "74A", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "96A", "96B", "96A", "96B", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "122A", "123", "124", "125", "126", "127", "127A", "127B", "128", "129", "130", "131", "132", "133", "133A", "134", "135", "136", "137", "138", "139", "140", "141", "142", "143", "144", "144A", "145", "146", "146A", "146B", "147", "148", "149", "149A", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "173A", "173B", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "188A", "188B", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "204A", "205", "206", "207", "208", "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "218A", "219", "220", "221", "222", "223", "223A", "224", "225", "226", "227", "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "245A", "245B", "245C"]
    rat_trypsin = np.array([c for c in rat_trypsin])[red_sector]
    nums = np.array([c for c in nums])[red_sector]

    print(''.join(rat_trypsin), nums)
    
#    for subaln_idx, subaln_file in enumerate(glob("./results/*/*.npy")):
#        print(f"\n{'='*60}")
#        print(f"Subalignment {subaln_idx}: {subaln_file}")
#        print(f"{'='*60}")
#        
#        model = np.load(subaln_file, allow_pickle=True).item()
#        J_ord, h_ord = get_params_gauge(model['J'], model['h'])
#        
#        aa_array = np.array([a for a in AA_ALPHABET])
#        aa_to_idx = {aa:i for i, aa in enumerate(AA_ALPHABET)}
#        
#        # Position 10 corresponds to residue 189
#        pos_189 = 10
#        
#        # Start with original sequence
#        previous_seq = rat_trypsin.copy()
#        
#        # Make the D189S mutation
#        print(f"\nInitial mutation: D{nums[pos_189]}S")
#        current_seq = previous_seq.copy()
#        current_seq[pos_189] = 'S'
#        
#        # Perform 4 iterations to find compensatory mutations
#        for iteration in range(4):
#            print(f"\nIteration {iteration + 1}:")
#            
#            # Compute deltaE for previous sequence
#            deltaEs_previous = compute_all_single_mutations_deltaE(previous_seq, h_ord, J_ord, AA_ALPHABET)
#            
#            # Compute deltaE for current sequence
#            deltaEs_current = compute_all_single_mutations_deltaE(current_seq, h_ord, J_ord, AA_ALPHABET)
#            
#            # deltaDeltaE = current - previous
#            deltaDeltaE = deltaEs_current - deltaEs_previous
#            
#            # Find the best mutation
#            residue_idx, aa_idx = get_best_indices(deltaDeltaE)
#            
#            previous_aa = current_seq[residue_idx]
#            next_aa = aa_array[aa_idx]
#            residue_num = nums[residue_idx]
#            
#            print(f"  Best compensatory mutation: {previous_aa}{residue_num}{next_aa}")
#            print(f"  ΔΔE = {deltaDeltaE[residue_idx, aa_idx]:.4f}")
#            
#            # Update for next iteration
#            previous_seq = current_seq.copy()
#            current_seq = current_seq.copy()
#            current_seq[residue_idx] = next_aa
#            
#        print(f"\nFinal sequence after D{nums[pos_189]}S + 4 compensatory mutations:")
#        print(''.join(current_seq))
