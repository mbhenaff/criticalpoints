#!/bin/bash

for s in {1..100}; do
	qsub -v seed=$s,layers=2,hidden=5,nexp=100 submit_job.pbs
done





