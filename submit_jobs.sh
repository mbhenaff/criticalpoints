
#!/bin/bash

for h in 10 25 50; do
    for s in {1..500}; do	
	qsub -v dataset=reuters,seed=$s,layers=1,hidden=$h,nexp=2,weight_decay=0 submit_run_cl.pbs
    done
done




