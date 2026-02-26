#!/bin/bash

count=0
for f in ./data/RedSecMar/subaln_seq_*.npy; do
	python SBM-CM-family.py RedSecMar_SubAln_$count $f --TestTrain 0 --m 1 --k_MCMC 5000 --rep 1 --N_av 1 --N_iter 1000 --theta 0.15 --ParamInit zero --lambdJ 0.01 --N_chains 500
	((  count++ ))
done
