#!/bin/bash
#SBATCH --job-name=multiGPUtest  # create a short name for your job
#SBATCH --nodes=8                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4           	 # number of gpus per node
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)

module purge
# module load cudatoolkit/11.1 openmpi/cuda-11.1/gcc/4.1.1
# module load cudatoolkit/12.2 nvhpc/22.5 openmpi/nvhpc-22.5/4.1.3
module load cudatoolkit/12.4 nvhpc/24.5 openmpi/cuda-12.4/nvhpc-24.5/4.1.6

# export OMPI_COMM_WORLD_RANK=$SLURM_PROCID
# srun nsys profile --trace=mpi,cuda,nvtx -o dendrite.%q{OMPI_COMM_WORLD_RANK} ./dendrite -niter 10 -nx 512 -ny 512 -nz 512 -vtk

srun ./dendrite -niter 40000 -nccheck 100000 -nx 512 -ny 32768 -nz 512 # -vtk