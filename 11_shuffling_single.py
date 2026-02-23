# Single-step shuffling training for SLURM parallelization (sector subalignments)
# Usage: python 11_shuffling_single.py <subaln_idx> <step_number>
#
# subaln_idx: 0-9 (which subalignment to use)
# step:
#   0: No shuffling (baseline)
#   1-23: Progressive sector column shuffling (23 columns, marion_red_sector)

import numpy as np
import argparse

import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
from pathlib import Path
import SBM

ROOT = Path(SBM.__file__).resolve().parents[2] 
results_dir = ROOT / ".." / "results"

# run_SBM function (from SBM-CM-family.py)
def run_SBM(Input_MSA, fam, Model='SBM', N_iter=1000, m=1, N_chains=500, Nb_av=10, k_MCMC=5000, 
            ParamInit='zero', lambdJ=0.01, lambdh=0, theta=0.15):
    fam = str(fam)
    
    W_rep = np.array([[]])
    Jnorm_rep = np.array([[]])
    Seeds_rep = np.zeros(Nb_av)
    Extime_rep = np.zeros(Nb_av)
    for n_av in range(Nb_av):
        print('AVG: ',n_av)
        align = Input_MSA
        print('Database size: ', align.shape)

        options = dict([('Model', Model),
                        ('N_iter', N_iter), ('N_chains', N_chains), ('m', m), 
                        ('skip_log', 1), ('theta', theta), ('k_MCMC', k_MCMC),
                        ('lambda_h', lambdh), ('lambda_J', lambdJ),
                        ('Pruning', False), ('Pruning Mask', None),
                        ('Param_init', ParamInit),
                        ('Test/Train', False), ('Train sequences', None),
                        ('Weights', None), ('SGD', None),
                        ('Seed', None), ('Zero Fields', False), 
                        ('Store Parameters', None)])

        output = sbm.SBM(align, options)
        
        J_out,h_out = ut.Zero_Sum_Gauge(output['J'],output['h'])
        W_out = ut.Wj(J_out,h_out)
        W_rep = np.concatenate((W_rep,np.expand_dims(W_out,axis=0)),axis=int((n_av==0)))

        Jnorm_rep = np.concatenate((Jnorm_rep,np.expand_dims(output['J_norm'],axis=0)),axis=int((n_av==0)))

        Seeds_rep[n_av] = output['options']['Seed']
        Extime_rep[n_av] = output['Execution time']

    W_av = np.mean(W_rep,axis=0)
    J_av,h_av = ut.Jw(W_av,output['options']['q'])
    output_av = {'J':J_av,'h':h_av,'W_all':W_rep,'Seeds':Seeds_rep,'Execution times':Extime_rep,'J_norm':Jnorm_rep,'align':output['align'],'Test':output['Test'],'Train':output['Train']}

    output_av['options0'] = {'Model':output['options']['Model'], 
                             'N_iter':output['options']['N_iter'],
                             'N_chains':output['options']['N_chains'],
                             'm':output['options']['m'],
                             'theta':output['options']['theta'],
                             'k_MCMC':output['options']['k_MCMC'], 
                             'lambda_h':output['options']['lambda_h'], 
                             'lambda_J':output['options']['lambda_J'],  
                             'Param_init':output['options']['Param_init']}

    output_av['options1'] = {'skip_log':output['options']['skip_log'],
                             'Pruning':output['options']['Pruning'],
                             'Pruning Mask':output['options']['Pruning Mask'],
                             'Test/Train':output['options']['Test/Train'],
                             'Train sequences':output['options']['Train sequences'],
                             'Weights':output['options']['Weights'],
                             'SGD':output['options']['SGD'],
                             'Seed':output['options']['Seed'],
                             'Zero Fields':output['options']['Zero Fields'],
                             'Store Parameters':output['options']['Store Parameters'],
                             'Learning_rate':output['options']['Learning_rate'],
                             'Pruning_perc':output['options']['Pruning_perc'],
                             'Shuffle Columns':output['options']['Shuffle Columns'],
                             'q':output['options']['q'],
                             'L':output['options']['L']}

    dossier = results_dir / fam
    dossier.mkdir(parents=True, exist_ok=True)

    r = 0
    file_name = fam
    key_list = sorted(output_av['options0'].keys())

    for k in key_list:
        file_name += f"_{k}{output_av['options0'][k]}"

    file_name += f"_N_Av{Nb_av}"

    path_result = dossier / f"{file_name}_R{r}.npy"

    while path_result.exists():
        r += 1
        path_result = dossier / f"{file_name}_R{r}.npy"
    np.save(path_result, output_av)
    
    return output_av

def shuffle_columns(alignment, columns_to_shuffle):
    """Shuffle specified columns independently in the alignment."""
    shuffled_aln = np.copy(alignment)
    for col in columns_to_shuffle:
        np.random.shuffle(shuffled_aln[:, col])
    return shuffled_aln

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single-step shuffling SBM training (sector subalignments)')
    parser.add_argument('subaln_idx', type=int, help='Subalignment index (0-9)')
    parser.add_argument('step', type=int, help='Step number (0-23)')
    parser.add_argument('--subaln_dir', type=str, default='./data/subalns', help='Subalignments directory')
    args = parser.parse_args()

    # Load subalignment (already sector-only, deduplicated, weighted sampled)
    subaln_file = f"{args.subaln_dir}/subaln{args.subaln_idx}_seq.npy"
    aln = np.load(subaln_file)
    M, L_sector = aln.shape
    print(f"Loaded subalignment {args.subaln_idx}: {M} sequences, {L_sector} positions")

    # Sector columns from 03_create_subalns.py (23 columns, marion_red_sector sorted)
    # marion_red_sector = [2, 21, 23, 88, 107, 164, 183, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 239]
    # These map to indices 0-22 in the subalignment

    # Sector columns ordered by increasing relevance (from 05_combine_mutations.py marion_red_sector reversed)
    marion_red_sector_sorted = [2, 21, 23, 88, 107, 164, 183, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 239]
    sector_to_idx = {col: idx for idx, col in enumerate(marion_red_sector_sorted)}
    
    # marion_red_sector in increasing relevance order (reversed from original decreasing order)
    marion_red_sector_by_relevance = [21, 23, 107, 222, 183, 88, 164, 231, 195, 229, 2, 194, 190, 228, 189, 200, 186, 227, 225, 237, 239, 197, 224]
    # Map to subalignment indices (0-22)
    sector_by_relevance = [sector_to_idx[col] for col in marion_red_sector_by_relevance]
    print(f"Shuffling order (subaln indices): {sector_by_relevance}")
    print(f"Number of sector columns to shuffle: {len(sector_by_relevance)}")

    # Set seed for reproducibility
    np.random.seed(42)

    # Parameters: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15

    step = args.step
    max_step = len(sector_by_relevance)  # 0 = no shuffle, 1-N = progressive shuffling

    if step > max_step:
        print(f"Error: step {step} is out of range (max step is {max_step})")
        exit(1)

    subaln_idx = args.subaln_idx

    if step == 0:
        # Step 0: no shuffling (baseline)
        print(f"SubAln {subaln_idx}, Step {step}: Training SBM with no shuffling...")
        run_SBM(aln, fam=f"SectorShuffling_SubAln{subaln_idx}_Step00_NoShuffle")

    else:
        # Steps 1+: progressive sector column shuffling
        sector_idx = step - 1
        columns_to_shuffle = sector_by_relevance[:sector_idx + 1]
        col_full = marion_red_sector_full[sector_idx]  # original column index for naming

        print(f"SubAln {subaln_idx}, Step {step}: Shuffling {sector_idx + 1}/{len(sector_by_relevance)} columns...")
        print(f"Columns shuffled (subaln indices): {columns_to_shuffle}")
        aln_shuffled = shuffle_columns(aln, columns_to_shuffle)
        run_SBM(aln_shuffled, fam=f"SectorShuffling_SubAln{subaln_idx}_Step{step:02d}_Col{col_full}")
