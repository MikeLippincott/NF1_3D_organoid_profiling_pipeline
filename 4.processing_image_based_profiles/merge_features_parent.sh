#!/bin/bash

patient=$1

echo "Processing patient: $patient"

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

parent_dir="${git_root}/data/$patient/extracted_features"
# get the list of all dirs in the parent_dir
dirs=$(ls -d "$parent_dir"/*)

python "$git_root"/4.processing_image_based_profiles/scripts/0.get_profiling_stats.py --patient "$patient"



# loop through each dir and submit a job
for dir in $dirs; do
    well_fov=$(basename "$dir")
    echo "$patient - $well_fov"
    # check that the number of jobs is less than 990
    # prior to submitting a job
    number_of_jobs=$(squeue -u "$USER" | wc -l)
    while [ "$number_of_jobs" -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u "$USER" | wc -l)
    done
    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=5:00 \
        --output=child_featurize-%j.out \
        "$git_root"/4.processing_image_based_profiles/merge_features_child.sh "$patient" "$well_fov"

done

conda deactivate

echo "All well_fov submitted for patient $patient"
