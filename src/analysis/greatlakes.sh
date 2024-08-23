#!/bin/bash
rm -rf fdr
git clone https://github.com/Davidvandijcke/fdr.git

scp -r "/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/code/fdd/src/" dvdijcke@greatlakes-xfer.arc-ts.umich.edu:/home/dvdijcke/fdr/src/


ssh dvdijcke@greatlakes.arc-ts.umich.edu
salloc --account=ffg0 --partition=gpu --time=3:00:00 --nodes=1 --gpus-per-node=1 --cpus-per-gpu=4 --mem-per-gpu=10g 

# reattach
# squeue -u dvdijcke # get job id
# scontrol show job $JOBID | grep NodeList # get node id
# srun --pty --jobid 55206671 -w gl1022 /bin/bash

sbatch --account=ffg0 --partition=spgpu --nodes=1 --gpus-per-node=6 --cpus-per-gpu=4 --mem-per-gpu=60g --time=03-00:00:00 --job-name=front_sure sbatch.sh

sbatch --account=ffg0 --partition=spgpu --nodes=1 --gpus-per-node=6 --cpus-per-gpu=4 --mem-per-gpu=60g --time=00-06:00:00 --job-name=front_sure sbatch.sh


srun --pty --jobid 57051692 -w gl1000 /bin/bash

sbatch --account=ffg0 --partition=spgpu --nodes=1 --gpus-per-node=6 --cpus-per-gpu=4 --mem-per-gpu=20g --time=05-00:00:00 --job-n
ame=front_slice_sure_quick sbatch.sh