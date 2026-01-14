import numpy as np
import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
import argparse
from pathlib import Path
import SBM

ROOT = Path(SBM.__file__).resolve().parents[3]
data_dir = ROOT / "data"
results_dir = ROOT / "data" / "results"

def run_SBM(Input_MSA,fam,Model,train_file,N_iter, m, N_chains_list,Nb_rep,Nb_av,k_MCMC,TestTrain,ParamInit,lambdJ,lambdh,theta,weights_file):
    fam = str(fam)

    for rep in range(Nb_rep):
        for N_chains in N_chains_list:
            W_rep = np.array([[]])
            Jnorm_rep = np.array([[]])
            Seeds_rep = np.zeros(Nb_av)
            Extime_rep = np.zeros(Nb_av)
            for n_av in range(Nb_av):
                print('AVG: ',n_av)
                align = np.load(str(Input_MSA))[:1000, :]
                print(align.shape)
                if train_file is not None:
                    ind_train = np.load(train_file)
                    print('Database size: ', align.shape, ' & Training set size: ', len(ind_train))
                else:
                    ind_train = None
                    print('Database size: ', align.shape)
#
#                if weights_file is not None:
#                    print(f"Loading pre-computed weights from: {weights_file}")
#                    weights = np.load(weights_file).ravel()[:1000]
#                    print(weights.shape)
#                    assert len(weights) == align.shape[0], \
#                        f"Weight length ({len(weights)}) != MSA size ({align.shape[0]})"
#                else:
                weights = None  # Will be computed during SBM

                options = dict([('Model', Model),
                                ('N_iter', N_iter), ('N_chains', N_chains), ('m', m),
                                ('skip_log', 1), ('theta', theta), ('k_MCMC', k_MCMC),
                                ('lambda_h', lambdh), ('lambda_J', lambdJ),
                                ('Pruning', False), ('Pruning Mask', None),
                                ('Param_init', ParamInit),
                                ('Test/Train', TestTrain==1), ('Train sequences', ind_train),
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

def main():
    # transform alignment in numpy array
    msa = ut.load_fasta("./data/iter_aln_dedup_sp.faa")
    np.save("./data/full_aln.npy", msa)
    print("About to run SBM...")

    # run SBM
    run_SBM(Input_MSA='./data/full_aln.npy',
            fam='SerProt',
            N_iter=400,
            N_chains_list=[500],
            m=1,
            theta=0.3,
            k_MCMC=100000,
            lambdh=0,
            lambdJ=0,
            ParamInit='Zero',
            TestTrain=0,
            Nb_rep=1,
            Nb_av=1,
            weights_file='./data/full_weights.npy',
            Model='SBM',
            train_file=None)

if __name__ == "__main__":
    main()
