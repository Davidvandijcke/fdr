#!/bin/bash

ssh dvdijcke@greatlakes.arc-ts.umich.edu
salloc --account=ffg0 --partition=gpu --time=01:00:00 --nodes=1 --gpus-per-node=3 --cpus-per-gpu=8 --mem-per-gpu=10g 