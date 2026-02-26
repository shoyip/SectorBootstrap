# The idea is the following:
# we want to take the sector.
# we start from the one with the sector and no shuffling.
# then we shuffle the rest.
# then we shuffle, one at a time, the sector columns, by decreasing relevance
# for each shuffling, we train an sbm model

# 1. import full alignment

# import libraries (from 03_create_subalns.py)
import numpy as np

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

# import NumPy arrays from .npz file (from 03_create_subalns.py)
aln_file = input("Enter alignment .npz filename [./data/full_aln.npz]: ")
if aln_file == "":
    aln_file = "./data/full_aln.npz"
aln = np.load(aln_file)

M, N = aln['seq'].shape

# enter the effective number of sequences (from 03_create_subalns.py)
M_eff = input("Enter effective number of sequences (M_eff) [17163]: ")
if M_eff == "":
    M_eff = 17163
else:
    M_eff = int(M_eff)

# enter the sector residues (from 05_combine_mutations.py marion_red_sector)
sector = input("Enter sector residues separated by commas [marion_red_sector/all/list]: ")
if sector == "":
    sector = [2, 21, 23, 88, 107, 164, 183, 186, 189, 190, 194, 195, 197, 200, 222, 224, 225, 227, 228, 229, 231, 237, 239]
elif sector == "all":
    sector = list(np.arange(0, N))
else:
    sector = list(map(int, sector.split(",")))
sector = sorted(sector)

# set the seed (from 03_create_subalns.py)
np.random.seed(42)

# choose subset of size M_eff (from 03_create_subalns.py)
subset_indices = np.random.randint(0, M, size=M_eff)

# mask the sequence array to produce new sub-alignment (from 03_create_subalns.py)
aln_seq_subset = letters_to_int(np.take(aln['seq'], subset_indices, axis=0))

# 2. train sbm on full alignment with subset of sequences

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

# 3. train sbm on full alignment with subset of sequences by shuffling the rest except the sector

def shuffle_columns(alignment, columns_to_shuffle):
    """Shuffle specified columns independently in the alignment."""
    shuffled_aln = np.copy(alignment)
    for col in columns_to_shuffle:
        np.random.shuffle(shuffled_aln[:, col])
    return shuffled_aln

# get non-sector columns
all_columns = set(range(N))
sector_set = set(sector)
non_sector_columns = sorted(all_columns - sector_set)

# Parameter set A: Nav=10, Nchains=50, Niter=400, kMCMC=100000, theta=0.3
# Parameter set B: Nav=10, Nchains=500, Niter=1000, kMCMC=5000, lambdaJ=0.01, theta=0.15

# Step 0: train model with no shuffling (baseline) - Parameter set A
print("Step 0: Training SBM on full alignment with no shuffling (param set A)...")
run_SBM(aln_seq_subset, fam="Shuffling_Step00_NoShuffle", 
        Nb_av=10, N_chains=50, N_iter=400, k_MCMC=100000, theta=0.3, lambdJ=0)

# Step 1: train model with shuffled non-sector columns - Parameter set A
print("Step 1: Training SBM with non-sector columns shuffled (param set A)...")
aln_shuffled_rest = shuffle_columns(aln_seq_subset, non_sector_columns)
run_SBM(aln_shuffled_rest, fam="Shuffling_Step01_RestOnly_ParamA", 
        Nb_av=10, N_chains=50, N_iter=400, k_MCMC=100000, theta=0.3, lambdJ=0)

# Step 2: train model with shuffled non-sector columns - Parameter set B
print("Step 2: Training SBM with non-sector columns shuffled (param set B)...")
run_SBM(aln_shuffled_rest, fam="Shuffling_Step02_RestOnly_ParamB", 
        Nb_av=10, N_chains=500, N_iter=1000, k_MCMC=5000, theta=0.15, lambdJ=0.01)

# train models by progressively shuffling sector columns (from 05_combine_mutations.py marion_red_sector)
# using Parameter set B
marion_red_sector = [21, 23, 107, 222, 183, 88, 164, 231, 195, 229, 2, 194, 190, 228, 189, 200, 186, 227, 225, 237, 239, 197, 224]
# filter to only include columns present in our sector
sector_by_relevance = [col for col in marion_red_sector if col in sector]

columns_shuffled_so_far = list(non_sector_columns)
for i, col in enumerate(sector_by_relevance):
    columns_shuffled_so_far.append(col)
    step = i + 3  # step 0 = no shuffle, step 1 = rest A, step 2 = rest B, step 3+ = sector columns
    print(f"Step {step}: Training SBM with sector column {col} shuffled ({i+1}/{len(sector_by_relevance)} sector cols, param set B)...")
    aln_shuffled = shuffle_columns(aln_seq_subset, columns_shuffled_so_far)
    run_SBM(aln_shuffled, fam=f"Shuffling_Step{step:02d}_Col{col}", 
            Nb_av=10, N_chains=500, N_iter=1000, k_MCMC=5000, theta=0.15, lambdJ=0.01)
