#!/bin/bash

module load miniforge
conda init bash
conda activate GFF_featurization

WELLFOV=$1
USEGPU=$2
PATIENT=$3

echo "Submitting jobs for $WELLFOV"
echo "Using GPU: $USEGPU"

number_of_jobs=$(squeue -u $USER | wc -l)
while [ $number_of_jobs -gt 950 ]; do
    sleep 1s
    number_of_jobs=$(squeue -u $USER | wc -l)
done

cd slurm_scripts || exit

if [ "$USEGPU" = "TRUE" ]; then
    echo "Running GPU version"

    sbatch \
        --nodes=1 \
        --ntasks=1 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=10:00 \
        --output=area_shape_gpu_child-%j.out \
        run_area_shape_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=30:00 \
        --output=colocalization_gpu_child-%j.out \
        run_colocalization_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=1:30:00 \
        --output=granularity_gpu_child-%j.out \
        run_granularity_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=2 \
        --partition=aa100 \
        --qos=normal \
        --gres=gpu:1 \
        --account=amc-general \
        --time=3:00:00 \
        --output=intensity_gpu_child-%j.out \
        run_intensity_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

else
    echo "Running CPU version"

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=area_shape_cpu_child-%j.out \
        run_area_shape_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=25 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=1:00:00 \
        --output=colocalization_cpu_child-%j.out \
        run_colocalization_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=20 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=8:00:00 \
        --output=granularity_cpu_child-%j.out \
        run_granularity_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

    sbatch \
        --nodes=1 \
        --ntasks=25 \
        --partition=amilan \
        --qos=normal \
        --account=amc-general \
        --time=6:00:00 \
        --output=intensity_cpu_child-%j.out \
        run_intensity_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"

fi

sbatch \
    --nodes=1 \
    --ntasks=1 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=10:00 \
    --output=neighbors_child-%j.out \
    run_neighbors_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"



sbatch \
    --nodes=1 \
    --ntasks=20 \
    --partition=amilan \
    --qos=normal \
    --account=amc-general \
    --time=24:00:00 \
    --output=texture_child-%j.out \
    run_texture_child.sh "$WELLFOV" "$USEGPU" "$PATIENT"


cd ../ || exit

echo "Featurization done"

