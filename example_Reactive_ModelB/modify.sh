#!/bin/bash

module purge
module load cudatoolkit/12.0

# Set the file name
file_name="Input.txt"

# Set the line number to change
line_number=25

# Define the start and end values
start_value=1e-5
end_value=2e-5
gap=1e-5
# values=("1e-4" "5e-5" "1e-5" "5e-6" "1e-6")
values=("5e-6" "1e-6" "5e-7" "1e-7" "5e-8" "1e-8" "5e-9" "1e-9" "5e-10" "1e-10")

# Create a temporary file
temp_file=$(mktemp)

n=1
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

    rm -r /scratch/gpfs/ruoyaoz/ModelB_DBA$n;
    mkdir /scratch/gpfs/ruoyaoz/ModelB_DBA$n;

    rm -rf CH
    make
    
    cp CH Input.txt modelB.cmd /scratch/gpfs/ruoyaoz/ModelB_DBA$n;
    cd /scratch/gpfs/ruoyaoz/ModelB_DBA$n;

    sbatch modelB.cmd;
    n=`expr $n + 1`;
	cd /home/ruoyaoz/ModelB_reaction/
done