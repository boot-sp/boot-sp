.. _User:

User
====

The user-mode software is `user_boot.py`.

The user mode is provided to demonstrate the concept of data-driven
stochastic programming so it provides full command line `Arguments`_.
However, in most serious applications, we expect that ``boot_sp.py`` will
be used as a callable library.

The main point of ``boot_sp.py`` is
bootstrap or bagging esimation of confidence intervals. Consequently, it
is expected the users will have other code to find a candidate solution
xhat. However, for aesthetic reasons, we offer an option to call
a function that uses ``mpi-sppy`` to compute xhat (see `xhat_creator`_).
