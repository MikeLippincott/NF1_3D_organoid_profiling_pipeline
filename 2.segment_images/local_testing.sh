#!/bin/bash

conda activate GFF_segmentation_nuclei

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb


a
cd scripts/ || exit 1
patient="NF0014_T1"
data_dir="/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/NF0014_T1/zstack_images/"
for well_fov in "$data_dir"*/; do
    well_fov=$(basename "$well_fov")
    input_subparent_name="zstack_images"
    mask_subparent_name="segmentation_masks"

    echo "Patient: $patient, WellFOV: $well_fov,  Input Subparent Name: $input_subparent_name, Mask Subparent Name: $mask_subparent_name"

    echo "Beginning segmentation for $patient - $well_fov"
    # python 0.nuclei_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01
    python run_each_segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name" --clip_limit 0.01

    # python 1.segmentation.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name"
    # python 1a.organoid_segmentation_derived_from_cell.py --patient "$patient" --well_fov "$well_fov" --input_subparent_name "$input_subparent_name" --mask_subparent_name "$mask_subparent_name"

done
cd ../ || exit 1


echo "All segmentation child jobs ran"

