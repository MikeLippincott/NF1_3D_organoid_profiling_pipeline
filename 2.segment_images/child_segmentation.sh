#!/bin/bash


# activate cellprofiler environment
# The following environment activation commands are commented out.
# Ensure the required environment is activated manually before running this script,
# or confirm that activation is handled by a parent script or workflow.
module load anaconda
conda init bash
conda activate GFF_segmentation

cd scripts/ || exit

well_fov=$1
patient=$2
echo "Processing well_fov $well_fov for patient $patient"
compartments=( "nuclei" "organoid" ) # we do not do 2.5D segmentation for cells in this script
# cells get segmented using a non 2.5D method rather a 3D method

python 0.segment_nuclei.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 3 \
    --clip_limit 0.05

python 2.segment_whole_organoids.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --window_size 4 \
    --clip_limit 0.1

for compartment in "${compartments[@]}"; do

    if [ "$compartment" == "nuclei" ]; then
        window_size=3
    elif [ "$compartment" == "organoid" ]; then
        window_size=4
    else
        echo "Not specified compartment: $compartment"

    fi
    python 2.segmentation_decoupling.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment" \
        --window_size "$window_size"

    python 3.reconstruct_3D_masks.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"

    python 4.post-hoc_mask_refinement.py \
        --patient "$patient" \
        --well_fov "$well_fov" \
        --compartment "$compartment"
done

python 5.segment_cells_watershed_method.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --clip_limit 0.05

python 4.post-hoc_mask_refinement.py \
    --patient "$patient" \
    --well_fov "$well_fov" \
    --compartment "cell"

python 6.post-hoc_reassignment.py \
    --patient "$patient" \
    --well_fov "$well_fov"

python 7.create_cytoplasm_masks.py \
    --patient "$patient" \
    --well_fov "$well_fov"

conda deactivate

cd ../ || exit

echo "Segmentation completed for well_fov $well_fov and patient $patient"
