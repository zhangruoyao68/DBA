#!/bin/bash

module purge
# module load cudatoolkit/11.1 openmpi/cuda-11.1/gcc/4.1.1
# module load cudatoolkit/12.2 nvhpc/22.5 openmpi/nvhpc-22.5/4.1.3
module load cudatoolkit/12.4 nvhpc/24.5 openmpi/cuda-12.4/nvhpc-24.5/4.1.6

make

# cd /home/ruoyaoz/
path="/scratch/gpfs/ruoyaoz/Reaction_cudampi_"

for n in 6; do
    echo $path$n
    rm -rf $path$n
    mkdir $path$n

    cp reaction cudampi.cmd $path$n

    #sed -i "/cd/c\cd /scratch/gpfs/ruoyaoz/DBA_cudampi_$n/" cudampi.cmd
    #sed -i "/srun/c\srun ./CH -niter 100000 -nccheck 20000 -nx 200 -ny 200 -nz 200 -c_init -0.35 -vtk" cudampi.cmd
    cd $path$n
    sbatch cudampi.cmd
done
