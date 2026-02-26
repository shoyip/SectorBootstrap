#!/bin/bash

count=0
for f in ./data/FullSubAln/subaln_seq_*.npy; do
	python SBM-CM-family.py FullSubAln_SubAln_$count $f --TestTrain 0 --m 1 --k_MCMC 100000 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero --lambdJ 0 --N_chains 50
	((  count++ ))
done
