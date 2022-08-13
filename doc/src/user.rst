.. _User:

User
====

The user-mode software is `user_boot.py`.

The user mode is provided to demonstrate the concept of data-driven
stochastic programming so it provides full command line :ref:Arguments` .
However, in most serious applications, we expect that ``boot_sp.py`` will
be used as a callable library. Programmers for such
an application may want to refer to ``user_boot.py`` for example code.

The main point of ``boot_sp.py`` is
bootstrap or bagging esimation of confidence intervals. Consequently, it
is expected the users will have other code to find a candidate solution
xhat. However, for aesthetic reasons, we offer an option to call
a function that uses ``mpi-sppy`` to compute xhat (see :ref:`xhat_generator`).
It will be called in user mode if the command line includes
``--xhat-fname "None"`` but if an ``npy`` files is given
for xhat, the `xhat_generator` function is not called.

The file ``boot-sp\examples\farmer.bash`` contains examples of user command lines.

help
^^^^

To see the list of commands for any particular module, give the command:

.. code-block:: bash

    $ python -m bootsp.user_boot Y --help

where Y is replaced by the module name. For example, if you cd to ``boot-sp/examples/farmer`` you can try

.. code-block:: bash

    $ python -m bootsp.user_boot Y --help

Note that you can give ``--known-optimal`` to the user_boot command line, but it won't do anything with it. In future releases, this
extraneous option will probably be removed.
