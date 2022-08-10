.. _Simulate:

Simulate
========

The simulation software is `simulate_boot.py`.

Overview
--------

The idea behind simulation mode is that a researcher has a problem with a known, or presumed, optimal solution. They may also have a candidate solution (xhat) that they have computed, but they could have it computed by the simulation software
if their module has proper support (see `optional`_).

xxx not meant for stand-alone running (even though it is runnable) xxx see paper_runs xxxx

xxxx steal text from paper

xxxx simcount

json file
---------

The module name is given on the command line and the rest of the arguments are given in a json file as described in the `commands`_ section. An example of such
a json file is ``boot-sp/paper_runs/famer.json``.


general_prep.py
---------------

This program prepares npy files used by simulations. xxxx
