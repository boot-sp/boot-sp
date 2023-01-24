.. _commands:

Commands
========

Running the programs
--------------------

The most general way is to use the ``python -m`` terminal command for the program for the desired mode.


User mode
^^^^^^^^^

The command for user mode is

.. code-block:: bash

   $ python -m bootsp.user_boot module arguments

where ``module`` is the name of a Python module (without `.py`, even
though the file name itself has `.py`) such as farmer that contains a
scenario creator with helper functions and ``arguments`` is list of
double-dash-intiated :ref:`Arguments`.
names, usually with an argument value
such as ``--solver-name cplex``. A fairly long list of arguments is
required in user mode, which is why most users put their command lines
in a shell script (e.g., a bash script such as ``farmer.bash``).


Simulation mode
^^^^^^^^^^^^^^^

The command for simulation mode is

.. code-block:: bash

   $ python -m bootsp.simulate_boot filename

where ``filename`` is the name of a Python such as `farmer.json` that contains a full :ref:`Arguments` set for the simulation.

.. _Arguments:

Arguments
---------

Simulation and user modes use almost the same arguments, but they are formatted a little
bit differently. In simulation mode, some of the arguments have underscores, but not dashes.
In user mode, some arguments have dashes, but none have underscores. For example, there
is a ``solver_name`` argument in simulation mode and a ``solver-name`` argument in user mode.

In simulation mode, the
argument values are given in a json file, while in user mode, they are given on the command line.
Here is a list of arguments giving the with the simulation mode version, the user mode version and
some discusion. In the json format, all string values are quote delimited.

*    ``max_count``, ``--max-count``: The total sample size given as an integer such as 100.

* ``module_name``, n/a: The name of of the python module that has the scenario creator and help functions given in the json file as a string such as "farmer". There is not command line argument name for this in user mode, where the module name is given as the first argument without an argument name.
     
* ``xhat_fname``, ``--xhat-fname``: When xhat (the estimated, or candidate) solution is computed by another program (which is common and recommended in simulation mode), this argument gives the name of an numpy file that has the solution as string such as "xhat.npy". If there is no file, the value should be the string "None". One way to create such a file is to use :ref:`boot_general_prep`

*     ``optimal_fname``, n/a: This gives the file name for an optimal (or presumed optimal) solution in a format written by ``mpi-sppy`` code. The name is given as as a string such as "schultz_optimal.npy" and is ignored in user mode. If the name "None" is given, the software will compute an estimated global optimum using ``max_count`` scenarios.  One way to create such a file is to use :ref:`boot_general_prep`.

* ``candidate_sample_size``, ``--candidate-sample-size``: The ``boot-sp`` software can call a function in the module to generate an xhat solution (see the :ref:`optional` section). This argument provides the sample size. It corresponds to the paramater M given in the paper. It is given as an intger such as 25.  If the ``xhat_fname`` argument is not "None", then ``candidate_sample_size`` is ignored.

*     ``sample_size``, ``--sample-size``: This value is the sample size used to for bootstrap or bagging. It corresponds to N in the paper and is given as an integer such as 75.  

*     ``subsample_size``, ``--subsample-size``: The subsample size used for bagging. It is given as an integer such as 10. It is ignored for bootstrap methods.

*     ``nB``, ``--nB``: The number of subsamples to take. It is given as an intger such as 10.

*     ``alpha``, ``--alpha``: significance level for the confidence intervals. It is given as a floating point number such as 0.05 for 95\% confidence.

*     ``seed_offset``, ``--seed-offset`` : This option is provided so that modelers who want to enable replication with difference seeds can do so. For some instances it can be used to assure independence between the psuedo-random number streams used to compute xhat and those used for confidence interval estimation. It is given as an integer. Unless you have a reason to do otherwise, just use 0, or, in user-mode, don't supply it.

*     ``solver_name``, ``--solver-name``: The name of the solver to be used given as a string such as "gurobi_direct".

*      ``trace_fname``, n/a: This is usually "None", but if it is not none the named file is opened in append mode and information about the simulation is written. Since the file is opned to append, it must already exist.

*       ``coverage_replications``, n/a: For simulations, this controls the number of replications used to computed coverage. It is an integer, e.g. 100.

*     ``boot_method``, ``--boot-method``: The method given as a string. Here are the choices (underscores in the string tokens are used in user and simulation mode):

    - "Classical_gaussian":  Classical boostrap using the Gussian to get confidence intervals [eichhorn2007stochastic]_
      
    - "Classical_quantile": Classical boostrap using the quantiles to get confidence intervals [eichhorn2007stochastic]_
      
    - "Extended": Extended bootstrap as described in [eichhorn2007stochastic]_

    - "Subsampling": A subsampling bootstrap mention briefly in [eichhorn2007stochastic]_

    - "Bagging_with_replacement": Bagging with replacement [lam2018assessing]_

    - "Bagging_without_replacement": Bagging without replacement [lam2018assessing]_


In addition to these arguments, there may be problem-specific arguments (e.g. "crops_multiplier" for
the scalable farmer problem).

Farmer Examples
---------------

For these two examples, cd to ``boot-sp/examples/farmer``.

simulate
^^^^^^^^

.. code-block:: bash

   $ python -m bootsp.simulate_boot farmer.json
   

user
^^^^

.. code-block:: bash

    $ python -m bootsp.user_boot farmer --max-count 121 --candidate-sample-size 1 --sample-size 75 --subsample-size 10 --nB 10 --alpha 0.05 --seed-offset 100  --solver-name cplex --boot-method Bagging_with_replacement --xhat-fname farmer_xhat.npy

Note that in this particular command ``--candidate-sample-size 1`` is ignored because a precomputed xhat is provided by ``--xhat-fname farmer_xhat.npy``
