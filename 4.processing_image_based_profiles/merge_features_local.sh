#!/bin/bash

module load anaconda
conda init bash
conda activate nf1_image_based_profiling_env

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient_array=( "NF0014" "NF0016" "NF0018" "NF0021" "SACRO219" )

cd scripts/ || exit

for patient in "${patient_array[@]}"; do

    python 0.get_profiling_stats.py --patient "$patient"
    # get the list of all dirs in the parent_dir
    parent_dir="../../data/$patient/extracted_features"
    # get the list of all dirs in the parent_dir
    dirs=$(ls -d $parent_dir/*)
    for dir in $dirs; do
        well_fov=$(basename $dir)
        echo $well_fov
        python 1.merge_feature_parquets.py --well_fov "$well_fov" --patient "$patient"
    done
done


cd ../ || exit

conda deactivate

echo "All features merged for patients" "${patient_array[@]}"
