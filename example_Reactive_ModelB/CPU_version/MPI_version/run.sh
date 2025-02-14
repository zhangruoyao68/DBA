#!/bin/bash

module purge
module load intel/19.1.1.217
module load intel-mpi/intel/2019.7

make

# cd /home/ruoyaoz/
path="/scratch/gpfs/ruoyaoz/Reaction_MPI_"

for n in 1; do
    echo $path$n
    rm -rf $path$n
    mkdir $path$n

    cp base_CPU Input.txt mpi.cmd $path$n

    cd $path$n
    sbatch mpi.cmd
done
