Code to estimate distributions of local minima. Many random weight initializations are chosen uniformly and the classifier is trained for a fixed number of epochs. The end solution is recorded as well as the value of the loss on the training and test sets. Computing a solution is considered 1 experiment. 

The script runCl.lua conducts a certain number of experiments (since each experiment is relatively short it is better for the same job to do many of them). Each call to runCl should be made with a different seed. 

Use submit_jobs.sh to call runCl and pass different arguments such as the number of layers, hidden units and number of experiments per job. 
