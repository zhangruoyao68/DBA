#!/bin/bash
#SBATCH --job-name=cxx_mpi       # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=16     # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
# SBATCH --mail-type=begin        # send email when job begins
# SBATCH --mail-type=end          # send email when job ends
# SBATCH --mail-type=fail         # send mail if job fails
# SBATCH --mail-user=ruoyaoz@princeton.edu

module purge
module load intel/19.1.1.217
module load intel-mpi/intel/2019.7
# module load intel-oneapi/2024.2
# module load intel-mpi/oneapi/2021.13

srun ./Base