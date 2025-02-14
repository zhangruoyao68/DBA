#!/bin/bash

module purge
module load cudatoolkit/12.0

# Set the file name
file_name="Input.txt"

# Set the line number to change
line_number=15

# Define the start and end values
start_value=1e-5
end_value=2e-5
gap=1e-5

values=("5e-5" "1e-5")
#values=("0")
# values=("5e-6" "1e-6" "5e-7" "1e-7" "5e-8" "1e-8" "5e-9" "1e-9" "5e-10" "1e-10")

# Create a temporary file
temp_file=$(mktemp)

n=11
#for new_value in $(seq $start_value $gap $end_value); do
for new_value in "${values[@]}"; do
    # Loop through the input file and replace the value on the specified line
    awk -v line_num="$line_number" -v new_val="$new_value" '
        NR == line_num {
            $1 = new_val
        }
        { print }
    ' "$file_name" > "$temp_file"

    # Move the temporary file to the original file
    mv "$temp_file" "$file_name"

    echo "Metric_eps has been changed to $new_value."

    rm -r /scratch/gpfs/ruoyaoz/Dendrite_3D_$n;
    mkdir /scratch/gpfs/ruoyaoz/Dendrite_3D_$n;

    rm -rf 3D_dendritic_growth
    nvcc -o 3D_dendritic_growth kernel.cu -std=c++11
    
    cp 3D_dendritic_growth Input.txt gpu_dendrite.cmd /scratch/gpfs/ruoyaoz/Dendrite_3D_$n;
    cd /scratch/gpfs/ruoyaoz/Dendrite_3D_$n;

    sbatch gpu_dendrite.cmd;
    n=`expr $n + 1`;
	cd /home/ruoyaoz/3D_dendritic_growth_PFM_SGPU/3D_dendritic_updated
done