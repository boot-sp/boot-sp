#!/bin/bash -l

# the time parameter could be important because boot_farmer can hang
# shiva has 48 "cpus" per node

#!/bin/bash -l
#SBATCH --job-name=bfarmer_slurm
#SBATCH --output=bfarmer_slurm.out
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=2  # I think this really means threads
#SBATCH --time=0-0:20:00
#SBATCH --nodelist=c[2]

# The next line may need to be edited.
# It sets of up the python environment (maybe you don't need it)
conda activate mpisppy

# export GRB_LICENSE_FILE=/home/dlwoodruff/software/gurobi950/licenses/c2/gurobi.lic

mpiexec -np ${SLURM_NTASKS} python boot_farmer.py bfarmer.json bfarmer_opt.txt

# remember stdout is buffered both by Python and by slurm
