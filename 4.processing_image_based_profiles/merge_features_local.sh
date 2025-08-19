#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi
jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/4.processing_image_based_profiles/scripts/ "$git_root"/4.processing_image_based_profiles/notebooks/*.ipynb

patient_array_file_path="$git_root/data/patient_IDs.txt"
# read the patient IDs from the file into an array
if [[ -f "$patient_array_file_path" ]]; then
    readarray -t patient_array < "$patient_array_file_path"
else
    echo "Error: File $patient_array_file_path does not exist."
    exit 1
fi

# setup the logs dir
if [ -d "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" ]; then
    rm -rf "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/"
fi
mkdir -p "$git_root/4.processing_image_based_profiles/logs/patient_well_fovs/" # create the patients directory if it doesn't exist


for patient in "${patient_array[@]}"; do

    python "$git_root"/4.processing_image_based_profiles/scripts/0.get_profiling_stats.py --patient "$patient"
    # get the list of all dirs in the parent_dir
    parent_dir="$git_root/data/$patient/extracted_features"
    # get the list of all dirs in the parent_dir
    dirs=$(ls -d "$parent_dir"/*)
    for dir in $dirs; do
        well_fov=$(basename "$dir")
        # check if the well fov is a run stats dir
        if [[ $well_fov == *"run_stats"* ]]; then
            continue
        fi
        echo "$patient - $well_fov"
        log_file="$git_root/4.processing_image_based_profiles/logs/${patient}_${well_fov}.log"
        touch "$log_file"  # create the log file if it doesn't exist
        {
            python "$git_root"/4.processing_image_based_profiles/scripts/1.merge_feature_parquets.py --patient "$patient" --well_fov "$well_fov"
            python "$git_root"/4.processing_image_based_profiles/scripts/2.merge_sc.py --patient "$patient" --well_fov "$well_fov"
            python "$git_root"/4.processing_image_based_profiles/scripts/3.organoid_cell_relationship.py --patient "$patient" --well_fov "$well_fov"
        } >> "$log_file" 2>&1
    done
    patient_log_file="$git_root/4.processing_image_based_profiles/logs/patients/${patient}.log"
    mkdir -p "$(dirname "$patient_log_file")"  # create the patients directory if it doesn't exist
    touch "$patient_log_file"  # create the patient log file if it doesn't exist
    {
        python "$git_root"/4.processing_image_based_profiles/scripts/5.combining_profiles.py --patient "$patient"
        python "$git_root"/4.processing_image_based_profiles/scripts/6.annotation.py --patient "$patient"
        python "$git_root"/4.processing_image_based_profiles/scripts/7.normalization.py --patient "$patient"
        python "$git_root"/4.processing_image_based_profiles/scripts/8.feature_selection.py --patient "$patient"
        python "$git_root"/4.processing_image_based_profiles/scripts/9.aggregation.py --patient "$patient"
        python "$git_root"/4.processing_image_based_profiles/scripts/10.merge_consensus_profiles.py --patient "$patient"
    } >> "$patient_log_file" 2>&1

done

python "$git_root"/4.processing_image_based_profiles/scripts/11.combine_patients.py

conda deactivate

echo "All features merged for patients" "${patient_array[@]}"
