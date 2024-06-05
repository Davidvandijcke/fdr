#!/bin/bash

ssh dvdijcke@greatlakes.arc-ts.umich.edu
salloc --account=ffg0 --partition=gpu --time=3:00:00 --nodes=1 --gpus-per-node=1 --cpus-per-gpu=4 --mem-per-gpu=10g 

# reattach
# squeue -u dvdijcke # get job id
# scontrol show job $JOBID | grep NodeList # get node id
# srun --pty --jobid 55206671 -w gl1022 /bin/bash

sbatch --account=ffg0 --partition=gpu --nodes=1 --gpus-per-node=1 --cpus-per-gpu=4 --mem-per-gpu=10g --time=00-04:00:00 --job-name=2D sbatch.sh

srun --pty --jobid 57051692 -w gl1000 /bin/bash