#!/bin/bash -l

#!/bin/bash -l
#SBATCH --job-name=nonunique_schultz
#SBATCH --output=nonunique_schultz_slurm.out
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2  # I think this really means threads
#SBATCH --nodelist=c[4]
#SBATCH --time="0-22:00:00"

# The next line may need to be edited.
# It sets of up the python environment (maybe you don't need it)
conda activate bootsp

# You need to edit or delete the next line:
export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c4/gurobi.lic

mpiexec -np ${SLURM_NTASKS} python simulate_experiments.py nonunique_schultz.json experiments.json

# remember stdout is buffered both by Python and by slurm
