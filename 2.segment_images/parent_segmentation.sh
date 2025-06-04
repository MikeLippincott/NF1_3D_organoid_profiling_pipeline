#!/bin/bash
# activate  cellprofiler environment
module load anaconda
conda init bash
conda activate GFF_segmentation

jupyter nbconvert --to=script --FilesWriter.build_directory=scripts/ notebooks/*.ipynb

patient=$1

echo "Processing patient $patient"

cd scripts/ || exit
# get all input directories in specified directory
z_stack_dir="../../data/$patient/zstack_images"
mapfile -t input_dirs < <(ls -d "$z_stack_dir"/*)
cd ../ || exit
total_dirs=$(echo "${input_dirs[@]}" | wc -w)
echo "Total directories: $total_dirs"
current_dir=0

touch segmentation.log
# loop through all input directories
for well_fov in "${input_dirs[@]}"; do
    number_of_jobs=$(squeue -u $USER | wc -l)
    while [ $number_of_jobs -gt 990 ]; do
        sleep 1s
        number_of_jobs=$(squeue -u $USER | wc -l)
    done
    well_fov=$(basename "$well_fov")
    current_dir=$((current_dir + 1))
    echo -ne "Processing directory $current_dir of $total_dirs\r"
    echo "Beginning segmentation for $well_fov"
    sbatch \
        --nodes=1 \
        --ntasks=6 \
        --partition=aa100 \
        --gres=gpu:1 \
        --qos=normal \
        --account=amc-general \
        --time=2:00:00 \
        --output=segmentation_child-%j.out \
        child_segmentation.sh "$well_fov" "$patient"

done

# deactivate cellprofiler environment
conda deactivate

echo "All segmentation child jobs submitted"

