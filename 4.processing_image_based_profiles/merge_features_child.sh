#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

well_fov=$1
patient=$2

cd scripts/ || exit

python 1.merge_feature_parquets.py --well_fov "$well_fov" --patient "$patient"

cd ../ || exit

conda deactivate

echo "Patient $patient well_fov $well_fov completed"
