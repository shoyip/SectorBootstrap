# Single-step shuffling training for SLURM parallelization
# Usage: python 10_shuffling_single.py <step_number> [--aln_file <path>]

# import libraries (from 03_create_subalns.py)
import numpy as np
import argparse

# import SBM modules (from SBM-CM-family.py)
import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
from pathlib import Path
import SBM

ROOT = Path(SBM.__file__).resolve().parents[2] 
results_dir = ROOT / ".." / "results"

# letters_to_int function (from 03_create_subalns.py)
def letters_to_int(aln, alphabet='-ACDEFGHIKLMNPQRSTVWY'):
    if isinstance(alphabet, str):
        alphabet = list(alphabet)
    else:
        raise ValueError("'alphabet' must be a string.")
    letter_to_int = {letter: i for i, letter in enumerate(alphabet)}
    return np.vectorize(letter_to_int.get)(aln)

# run_SBM function (from SBM-CM-family.py)
def run_SBM(Input_MSA, fam, Model='SBM', N_iter=400, m=1, N_chains=50, Nb_av=1, k_MCMC=100000, 
            ParamInit='zero', lambdJ=0, lambdh=0, theta=0.3):
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
    parser = argparse.ArgumentParser(description='Single-step shuffling SBM training')
    parser.add_argument('step', type=int, help='Step number (0-25)')
    parser.add_argument('--aln_file', type=str, default='./data/full_aln.npz', help='Alignment file path')
    parser.add_argument('--M_eff', type=int, default=17163, help='Effective number of sequences')
    args = parser.parse_args()

    # Load alignment
    aln = np.load(args.aln_file)
    M, N = aln['seq'].shape

    # Sector residues (from 05_combine_mutations.py marion_red_sector)
    sector = [2, 21, 23, 88, 107, 164, 183, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 239]

    # Set seed for reproducibility (from 03_create_subalns.py)
    np.random.seed(42)

    # Choose subset of size M_eff (from 03_create_subalns.py)
    subset_indices = np.random.randint(0, M, size=args.M_eff)

    # Convert to integer alignment (from 03_create_subalns.py)
    aln_seq_subset = letters_to_int(np.take(aln['seq'], subset_indices, axis=0))

    # Get non-sector columns
    all_columns = set(range(N))
    sector_set = set(sector)
    non_sector_columns = sorted(all_columns - sector_set)

    # Sector columns ordered by increasing relevance (from 05_combine_mutations.py marion_red_sector reversed)
    marion_red_sector = [21, 23, 107, 222, 183, 88, 164, 231, 195, 229, 2, 194, 190, 228, 189, 200, 186, 227, 225, 237, 239, 197, 224]
    sector_by_relevance = [col for col in marion_red_sector if col in sector]

    # Parameter set A: Nav=10, Nchains=50, Niter=400, kMCMC=100000, theta=0.3
    # Parameter set B: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15

    step = args.step

    if step == 0:
        # Step 0: no shuffling - Parameter set A
        print(f"Step {step}: Training SBM on full alignment with no shuffling (param set A)...")
        run_SBM(aln_seq_subset, fam="Shuffling_Step00_NoShuffle", 
                Nb_av=10, N_chains=50, N_iter=400, k_MCMC=100000, theta=0.3, lambdJ=0)

    elif step == 1:
        # Step 1: rest shuffled - Parameter set A
        print(f"Step {step}: Training SBM with non-sector columns shuffled (param set A)...")
        aln_shuffled = shuffle_columns(aln_seq_subset, non_sector_columns)
        run_SBM(aln_shuffled, fam="Shuffling_Step01_RestOnly_ParamA", 
                Nb_av=10, N_chains=50, N_iter=400, k_MCMC=100000, theta=0.3, lambdJ=0)

    elif step == 2:
        # Step 2: rest shuffled - Parameter set B
        print(f"Step {step}: Training SBM with non-sector columns shuffled (param set B)...")
        aln_shuffled = shuffle_columns(aln_seq_subset, non_sector_columns)
        run_SBM(aln_shuffled, fam="Shuffling_Step02_RestOnly_ParamB", 
                Nb_av=10, N_chains=500, N_iter=1000, k_MCMC=5000, theta=0.15, lambdJ=0.01)

    else:
        # Steps 3+: progressive sector column shuffling - Parameter set B
        sector_idx = step - 3
        if sector_idx >= len(sector_by_relevance):
            print(f"Error: step {step} is out of range (max step is {len(sector_by_relevance) + 2})")
            exit(1)

        # Columns to shuffle: non-sector + sector columns up to current
        columns_to_shuffle = list(non_sector_columns) + sector_by_relevance[:sector_idx + 1]
        col = sector_by_relevance[sector_idx]

        print(f"Step {step}: Training SBM with sector column {col} shuffled ({sector_idx + 1}/{len(sector_by_relevance)} sector cols, param set B)...")
        aln_shuffled = shuffle_columns(aln_seq_subset, columns_to_shuffle)
        run_SBM(aln_shuffled, fam=f"Shuffling_Step{step:02d}_Col{col}", 
                Nb_av=10, N_chains=500, N_iter=1000, k_MCMC=5000, theta=0.15, lambdJ=0.01)
