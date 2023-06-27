#!/bin/bash

conda activate pytorch
rm -rf fdd
git clone https://github.com/Davidvandijcke/fdd.git
cd fdd
pip install .
cd src/analysis
python simulations_2d.py