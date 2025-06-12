#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env


patient=$1

parent_dir="../data/$patient/extracted_features"
# get the list of all dirs in the parent_dir
dirs=$(ls -d $parent_dir/*)

cd scripts/ || exit
python 0.get_profiling_stats.py --patient "$patient"
cd ../ || exit


# loop through each dir and submit a job
for dir in $dirs; do
    well_fov=$(basename $dir)
    echo $well_fov
    # check that the number of jobs is less than 990
    # prior to submitting a job
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=long \
        --account=amc-general \
        --time=1:00:00 \
        --output=parent_featurize-%j.out \
        merge_features_parent.sh "$well_fov" "$patient"

done

conda deactivate

echo "All well_fov submitted for patient $patient"
