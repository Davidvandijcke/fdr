#!/bin/bash

conda activate pytorch
rm -rf fdd
git clone https://github.com/Davidvandijcke/fdd.git
cd fdd
pip install .
