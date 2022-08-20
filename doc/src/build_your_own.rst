.. _build:


Create a new instance
=====================

Most users want to copy one of the pre-built examples and modify it to create a new instance. The documentation provided here
helps with understanding that and also provides information for users who want to start from scratch.
An instance is defined by a ``Python module``, which is usually a file with a ``.py`` filename extension.
The file ``boot-sp/examples/lam_cvar.py`` is a simple example and the file ``boot-sp/examples/farmer.py`` is a more complicated example.


`scenario_creator` function
---------------------------

This function instantiates models for scenarios and usually attaches
some information about the scenario tree. Its first argument must be the scenario name. The other
two arguments are optional.

The `scenario_creator_kwargs` option specifies data that is
passed through from the calling program.
`scenario_creator_kwargs` must be a dictionary, and might give, e.g., a data
directory name or probability distribution information.  The
`scenario_creator` function returns an instantiated model for the
instance. I.e., it either creates a `ConcreteModel` or else it creates
and instantiates an `AbstractModel`.

The `scenario_creator` function somehow needs to create a list of
non-leaf tree node objects that are constructed by calling
`scenario_tree.ScenarioNode` which is not very hard for two stage
problems, because there is only one non-leaf node and it must be
called "ROOT".  
Node list entries can be entered individually, by adding an entire
variable implicitly including all index values, and/or by using wildcards. This is
illustrated in the ``mpi-sppy`` netdes example:

::
   
   # Add all indexes of model.x
   sputils.attach_root_node(model, model.FirstStageCost, [model.x, ])

::
   
   # Add all index of model.x using wild cards
   sputils.attach_root_node(model, model.FirstStageCost, [model.x[:,:], ])

The scenario probability should be attached by `scenario_creator` as
``_mpisppy_probability``. However, if you don't attach it, the scenarios are
assumed to be equally likely.

  See the ``mpi-sppy`` documenation for more information about this function.

EF Supplement List
^^^^^^^^^^^^^^^^^^

Advanced topic: The function ``attach_root_node`` takes an optional argument ``nonant_ef_suppl_list`` (that is passed through to the ``ScenarioNode`` constructor). This is a list similar to the nonanticipate Var list. These variables will not be given
multipliers by algorithms such as PH, but will be given non-anticipativity
constraints when an EF is formed, either to solve the EF or when bundles are
formed. For some problems, with the appropriate solver, adding redundant nonanticipativity constraints
for auxilliary variables to the bundle/EF will result in a (much) smaller pre-solved model.

kw_creator
----------

Uses the configuration object to return the keyword arguments (if there are any) for the scenario creator



scenario_names_creator
----------------------

Return the full list of num_scens scenario names. It has an option start at a scenario number other than zero.
        


inparser_adder
--------------

Add options unique to to the instance to the configuration object.


