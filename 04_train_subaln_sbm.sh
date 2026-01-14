#!/bin/bash

for each 
python SBM-CM-family.py SerProt data/full_aln.npy --TestTrain 0 --m 1 --rep 1 --N_av 1 --N_iter 400 --theta 0.3 --ParamInit zero --lambdJ 0 --lambdJ 0 --N_chains 70
