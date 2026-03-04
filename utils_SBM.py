import numpy as np
import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
from pathlib import Path
import SBM

current_folder = Path(__file__).resolve().parent
results_dir = current_folder / "models"

def run_SBM(Input_MSA, fam, weights=None, Model='SBM', N_iter=1000, m=1, N_chains=500, Nb_av=10, k_MCMC=5000,
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
                        ('Weights', weights), ('SGD', None),
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
