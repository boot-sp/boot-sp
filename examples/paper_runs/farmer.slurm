#!/bin/bash -l

# the time parameter could be important because boot_farmer can hang
# shiva has 48 "cpus" per node

#!/bin/bash -l
#SBATCH --job-name=farmer_slurm
#SBATCH --output=farmer_slurm.out
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2  # I think this really means threads
#SBATCH --nodelist=c[3]


# The next line may need to be edited.
# It sets of up the python environment (maybe you don't need it)
conda activate mpisppy

# export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c2/gurobi.lic

mpiexec -np ${SLURM_NTASKS} python simulate_experiments.py farmer.json experiments.json

# remember stdout is buffered both by Python and by slurm
