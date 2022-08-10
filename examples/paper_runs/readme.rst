paper_runs
==========

This directory contains files needed to do the simulation experiments in the paper. The slurm scripts can also be run as bash scripts, after editing.

You probably need to edit them
to remove (or alter) the conda activate line.

If they are used as bash scripts, you will need to replace ``${SLURM_NTASKS}`` with a number. Use a number that is about half the number of CPUs on your computer.  These simulations are intended to be run with MPI and take a lot of computing.  The scripts are designed to do all the simulations for one table in the paper.
