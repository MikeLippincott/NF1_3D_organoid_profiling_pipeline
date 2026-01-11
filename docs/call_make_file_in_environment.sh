#!/bin/bash

conda activate NF1_3D_featurization_sphinx_env

make clean
make html

conda deactivate

