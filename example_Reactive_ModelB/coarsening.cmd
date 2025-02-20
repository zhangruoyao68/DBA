#!/bin/bash
#SBATCH --job-name=size_dist       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=1:01:00          # total run time limit (HH:MM:SS)
# SBATCH --mail-type=all          # send email on job start, end and fail
# SBATCH --mail-user=ruoyaoz@princeton.edu

./a.out