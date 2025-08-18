#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=10:00
#SBATCH --output=merge_features_grand_parent-%j.out

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
if [ -d "$git_root/4.processing_image_based_profiles/logs" ]; then
    rm -rf "$git_root/4.processing_image_based_profiles/logs"
fi
mkdir "$git_root/4.processing_image_based_profiles/logs"


for patient in "${patient_array[@]}"; do
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
    --time=1:00:00 \
    --output=merge_features_parent-%j.out \
    "$git_root"/4.processing_image_based_profiles/merge_features_parent.sh "$patient"
done

conda deactivate

echo "All patients submitted for segmentation"
