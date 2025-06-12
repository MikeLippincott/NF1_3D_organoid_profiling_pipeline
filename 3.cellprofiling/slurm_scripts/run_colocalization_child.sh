#!/bin/bash
well_fov=$1
use_GPU=$2
patient=$3

echo "Running featurization for $patient $well_fov"
module load miniforge
conda init bash
conda activate GFF_featurization

cd ../scripts/ || exit

# start the timer
start_timestamp=$(date +%s)
if [ "$use_GPU" = "TRUE" ]; then
    echo "Running GPU version"
    python colocalization_gpu.py --well_fov $well_fov --patient $patient
else
    echo "Running CPU version"
    python colocalization.py --well_fov $well_fov --patient $patient
fi

end=$(date +%s)
echo "Time taken to run the featurization: $(($end-$start_timestamp))"

cd ../ || exit

# conda deactivate
