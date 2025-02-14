#!/bin/bash
#SBATCH --job-name=inspire-ground-irrad         # Job name
#SBATCH --output=dask_job_%j.log    # Standard output (%j for job ID)
#SBATCH --error=dask_job_%j.err     # Standard error (%j for job ID)
#SBATCH --time=02:00:00             # Total run time (hh:mm:ss)
#SBATCH --partition=shared          # Queue/partition to use
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks
#SBATCH --cpus-per-task=1           # CPUs per task
#SBATCH --mem=80G                   # Memory for the job
#SBATCH --account=inspire           # Account name


# i deleted the ntasks here but this might not be required with the slurmrunnner
# define number of tasks
# 2 extra: 1 for scheduler, etc.




# we will define everything early
# n tasks will give us n workers

module load anaconda3  # kestrel module name
# source activate rpp   # Activate your Anaconda environment
source activate /home/tford/.conda-envs/rpp

# run with configurations
python /home/tford/dev/PVDegradationTools/scripts/batch/inspire.py 03

conda deactivate
