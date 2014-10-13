#!/bin/bash

for h in 2 3 5 10 100; do
    for s in {1..100}; do
	qsub -v dataset=mnist,seed=$s,layers=1,hidden=$h,nexp=10 submit_compute_results.pbs
    done
done




