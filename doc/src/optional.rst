.. _optional:


Optional module functions
=========================

These functions are needed for xhat creation and finding an assumed optimal; they do not need to be supplied if they are created
elsewhere (the assumed optimal is not needed for user mode). They can be created elsewhere using the ``boot_general_prep.py`` program,
for example.

.. _xhat_generator:

xhat_generator
--------------

.. Note:

   Not all examples have an ``xhat_generator`` function.
   
The xhat generator function is assumed to have the name
``xhat_generator_I`` where I is replaced by the module name, e.g.,
``xhat_generator_farmer``.  The first argument is a list of scenario names, the second argument is the solver name, the third
argument is solver options and subsequent arguments are problem dependent and match the problem dependent arguments
for the ``scenario_creator`` function. Here is the function signature for the farmer example:

.. code-block: python
   
    def xhat_generator_farmer(scenario_names, solver_name="gurobi", solver_options=None, crops_multiplier=1, use_integer=False):


scenario_denouement
-------------------

This function, if present, is called by ``mpi-sppy`` for each scenario at termination and will usually not be called by ``boot-sp``. For most
applications, even in ``mpi-sppy``, the function simply contains `pass`.

