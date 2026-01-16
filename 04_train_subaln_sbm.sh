#!/bin/bash

count=0
for f in ./data/RedSecMar/subaln_seq_*.npy; do
	python SBM-CM-family.py RedSecMar_SubAln_$count $f --TestTrain 0 --m 1 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero --lambdJ 0 --lambdJ 0 --N_chains 50
	((  count++ ))
done
