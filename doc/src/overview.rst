.. _Overview:

Overview
========

Stochastic programming concerns optimization when the input data are
uncertain.  Many research articles concerning stochastic programming
begin with the assumption that the uncertain data have a distribution
that is known.  This software uses only sampled data to obtain both a
consistent sample-average solution and a consistent estimate of
confidence intervals for the optimality gap. The underlying
distribution whence the samples come is not required.  See [ThePaper]_ on OOL
for more details.

At first, the software is expected mainly to be used by researchers
and is only tested on Unix platforms. It might work on Windows, but is
not tested there. This documentation assumes you are using a Unix
shell such as Bash.


Roles
-----

We take that view that there are the following roles (that might be
filled by one person, or multiple people):

- *Modeler*: creates the Pyomo model and establishes the nature of the scenario tree.
- *User*: runs a program to get the results of optimization under uncertainty along with confidence intervals.
- *Researchers*: runs simulations to learn more about confidence interval methods.`
- *Contributors to boot-sp*: the creators of, and contributors to, ``boot-sp``.

Basics
------

The ``boot-sp`` software relies on a ``Pyomo`` model to define the underlying problem (i.e., a deterministic scenario) and relies
on ``mpi-sppy`` for some low level functions. Both of these packages must be installed.

The ``boot-sp`` software has two modes: ``simulation`` mode for researchers and ``user`` mode for modelers and end-users who have
data and a problem and want, or have, a solution and want confidence intervals.


