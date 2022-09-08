.. _Installation:

Installation and Quick Start
============================

This is a terminal application.

Installation
------------

#. Verify that a Python version 3.8 or higher is installed.

#. Verify that `git <https://github.com/>`_ is installed 

#. Install a solver such as ``cplex``, ``glpk``, ``gurobi``, ``ipopt`` or etc. so that is can be run from the command line.

#. Install `Pyomo <http://www.pyomo.org/>`_.

#. Install `mpi-sppy <https://github.com/Pyomo/mpi-sppy>`_ using a github clone.
   
#. cd to the directory where you want to put `boot-sp` and give these commands:

   
.. code-block:: bash

   $ git clone https://github.com/boot-sp/boot-sp.git
   $ cd boot-sp
   $ python setup.py develop

   
For parallel operation, you will need to install MPI as described in the next section.


Quick Install
-------------

This is not the recommend way to install, but if you are really in a
hurry, you have Python version 3.8 or higher installed, and you have
git installed, the following sequence of *nix shell commands might
install everytying else:

.. code-block:: bash

    $ pip install cplex
    $ pip install pyomo
    $ pip install mpi-sppy
    $ git clone https://github.com/boot-sp/boot-sp.git
    $ cd boot-sp
    $ python setup.py develop
    $ cd ..

For parallel operation, you will need to install MPI as described in the next section.

   
Quick Start
-----------

If you want to use the quick-start instructions exactly as written, you will need to install cplex. To install the community edition of
cplex, use

.. code-block:: bash

   $ pip install cplex

However, you can use other solvers by editing the file `boot-sp/examples/farmer.bash` to replace `cplex` with a different solver name that
has been installed.
   

Connect to the `boot-sp\examples directory` and give the terminal command:

.. code-block:: bash

   $ bash farmer.bash

to see the program user-mode program execute for the ``farmer`` problem [birge2011]_.

