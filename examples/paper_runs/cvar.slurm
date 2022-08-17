#!/bin/bash -l

#!/bin/bash -l
#SBATCH --job-name=cvar_slurm
#SBATCH --output=cvar_slurm.out
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1  # I think this really means threads
#SBATCH --nodelist=c[2]

# The next line may need to be edited.
# It sets of up the python environment (maybe you don't need it)
conda activate bootsp

# export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c2/gurobi.lic

mpiexec -np ${SLURM_NTASKS} python simulate_experiments.py cvar.json experiments.json

# remember stdout is buffered both by Python and by slurm
