#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

well_fov=$1
use_GPU=$2
patient=$3

echo "Running featurization for $patient $well_fov"

cd ../scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
python texture.py --well_fov $well_fov --patient $patient
end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

# conda deactivate
