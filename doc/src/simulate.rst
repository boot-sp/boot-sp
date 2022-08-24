.. _Simulate:

Simulate
========

The simulation software is `simulate_boot.py`.

Overview
--------

The idea behind simulation mode is that a researcher has a problem
with a known, or presumed, optimal solution. They may also have a
candidate solution (xhat) that they have computed, but they could have
it computed by the simulation software if their module has proper
support (see :ref:`optional`).

The program can run from the command line and takes a json file for an instance as its argument (e.g. ``farmer.py``).
However, it is mainly intended to be called in a loop from a script like ``simulate_experiments.py``
(see :ref:`simulate_experiments.py`).


json file
---------

The module name is given on the command line and the rest of the
arguments are given in a json file as described in the :ref:`commands`
section. An example of such a json file is
``boot-sp/paper_runs/famer.json``.

.. _boot_general_prep:


boot_general_prep
-----------------

The ``boot_general_prep.py`` program prepares two npy files used by simulations. It takes as its only argument a json file for an instance (e.g. ``farmer.json``)

.. code-block:: bash

   $ python -m bootsp.boot_general_prep farmer.json

The program outputs one file with xhat and another with an assumed optimal
create by solving the extensive form directly for ``max_count`` scenarios as
specified in the json file.
