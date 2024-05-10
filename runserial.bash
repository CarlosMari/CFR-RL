#!/bin/bash
#SBATCH -J launcher-test            # job name
#SBATCH -o launcher.o%j             # output and error file name (%j expands to SLURM jobID)
#SBATCH -N 1                        # number of nodes requested
#SBATCH -n 3                        # total number of tasks to run in parallel
#SBATCH -p development              # queue (partition) 
#SBATCH -t 00:30:00                 # run time (hh:mm:ss) 
#SBATCH --mail-type=all             # Send email at begin and end of job
#SBATCH -A MLL                      # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=carlosmari@utexas.edu

module load launcher
module load python3/3.9.2
source ../python-envs/CFR-RL/bin/activate

export OMP_NUM_THREADS=15

export LAUNCHER_WORKDIR=/scratch1/09701/cmari/test
export LAUNCHER_JOB_FILE=jobrun 

${LAUNCHER_DIR}/paramrun