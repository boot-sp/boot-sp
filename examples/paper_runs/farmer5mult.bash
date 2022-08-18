#!/bin/bash -l

# slurm directives are comments in a bash script
#SBATCH --job-name=farmer_slurm
#SBATCH --output=farmer_slurm.out
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2  # I think this really means threads
#SBATCH --nodelist=c[3]

mpiexec -np 1 python simulate_experiments.py farmer5mult.json experiments.json

