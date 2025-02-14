#!/bin/bash
#SBATCH --job-name=single-gpu     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=71:59:00          # total run time limit (HH:MM:SS)

# sends mail when process begins, and
# when it ends. Make sure you define your email
# address.
# SBATCH --mail-type=begin
# SBATCH --mail-type=end
# SBATCH --mail-user=ruoyaoz@princeton.edu

module purge
module load cudatoolkit/12.0

./CH