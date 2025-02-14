#!/bin/bash

module purge
module load intel/19.1.1.217
module load intel-mpi/intel/2019.7

make

# cd /home/ruoyaoz/
path="/scratch/gpfs/ruoyaoz/KHI_MPI_"

for n in 1; do
    echo $path$n
    rm -rf $path$n
    mkdir $path$n

    cp KH_CPU Input.txt mpi.cmd $path$n

    #sed -i "/cd/c\cd /scratch/gpfs/ruoyaoz/DBA_cudampi_$n/" cudampi.cmd
    #sed -i "/srun/c\srun ./CH -niter 100000 -nccheck 20000 -nx 200 -ny 200 -nz 200 -c_init -0.35 -vtk" cudampi.cmd
    cd $path$n
    sbatch mpi.cmd
done
