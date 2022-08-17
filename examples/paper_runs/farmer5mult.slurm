#!/bin/bash -l

# shiva has 48 "cpus" per node

#!/bin/bash -l
#SBATCH --job-name=farmer5mult_slurm
#SBATCH --output=farmer5mult_slurm.out
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=2  # I think this really means threads
#SBATCH --nodelist=c[3]


# The next line may need to be edited.
# It sets of up the python environment (maybe you don't need it)
conda activate bootsp

export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c3/gurobi.lic

mpiexec -np ${SLURM_NTASKS} python simulate_experiments.py farmer5mult.json experiments.json

# remember stdout is buffered both by Python and by slurm
