#!/bin/bash

module purge
module load cudatoolkit/12.0

n=1

for f in 1; do
    echo $n $f;

    rm -r /scratch/gpfs/ruoyaoz/ModelB_noDBA$n;
    mkdir /scratch/gpfs/ruoyaoz/ModelB_noDBA$n;

    rm -rf CH
    make
    
    cp CH Input.txt modelB.cmd /scratch/gpfs/ruoyaoz/ModelB_noDBA$n;
    cd /scratch/gpfs/ruoyaoz/ModelB_noDBA$n;

    sbatch modelB.cmd;
    n=`expr $n + 1`;
	cd /home/ruoyaoz/ModelB_reaction/
done