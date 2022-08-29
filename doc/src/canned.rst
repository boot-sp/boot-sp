.. _canned:

Pre-built Examples
==================

We provide examples with the ``boot-sp`` distribution and more are being added all the time. Here is a list
of some of the examples with comments about them:

* farmer: This is a widely used example from Birge and Louveaux [birge2011]_ extended to be scalable.

* lam_cvar: A very simple CVaR-only model that was used for experiments in [lam2018assessing]_.

* unique_schultz: A small example shown on page 129 of [eichhorn2007stochastic]_

* nonunique_schultz: A small example shown on page 131 of [eichhorn2007stochastic]_


paper_runs
^^^^^^^^^^

This directory contains files needed to do the simulation experiments
in the paper. The slurm scripts can also be run as bash scripts, after
editing. For one thing, you probably need to edit them
to remove (or alter) the conda activate line.

If they are used as bash scripts, you will need to replace
``${SLURM_NTASKS}`` with a number. Use a number that is about half the
number of CPUs on your computer if you are using MPI.  If you are not
using MPI, you will need to remove the mpiexec part of the command
line entirely. These simulations are intended to be run with MPI and
take a lot of computing.  The slurm scripts are designed to do all the
simulations for one table in the paper.


.. _simulate_experiments.py:

``simulate_experiments.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This Python program in ``paper_runs`` reads two json files that control the experiments that
are run for the paper that describes this software. It also serves
as a starting point to copy if you want to create your own simulation
program. Note that the definitions of N, nB, and k are in the paper.

The first json file is for the problem instance (e.g. ``farmer.json``) and
the second json program provides the values that will override the options
in the first file to form the experiments; the ``experiments.json`` file is an example.

The second json (e.g. ``experiments.json``) has three things:

* ``problem_sizelist``: This is a dictionary with a key for each problem definition module you want to run (e.g. ``unique_schultz``) and list of N values for that problem. The program ``simulate_experiments.py`` will assume that there is a module with that name (e.g., ``unique_shultz.py`` in the current directory). The program ``simulate_experiments.py``  will loop over the problems and for each, will loop over the values for N.

* ``nB_list``:  For each problem definition module and for each value N, ``simulate_experiments.py`` will loop over this list of nB values.  If you copy ``simulate_experiments.py`` to create your own program, you might want to make this a dictionary of lists so nB can vary wiht the problem.

* ``method_kfac``:  For each problem, each N, and each nB, each method that is a key for this dictionary will be simulated and for each of
  those, the values in the list will be muplied by $N$ to get the value for k. For the bootstrap methods, k is ignored (in some sense, it
  would be 1, but the value k is not used by the software).

Note the the first json file (e.g. ``farmer.json``) gives the name of the file with the (presumed) optimal solution.
