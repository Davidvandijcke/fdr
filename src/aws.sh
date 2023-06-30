#!/bin/bash

conda activate pytorch
rm -rf fdd
git clone https://github.com/Davidvandijcke/fdd.git
cd fdd
pip install .
cd src/analysis
nohup python simulations_2d.py

# # Check on Background Processes:
#     ps -eo pid,pgid,tpgid,args | awk 'NR == 1 || ($3 != -1 && $2 != $3)'
#     ps aux | grep python