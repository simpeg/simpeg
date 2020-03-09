.. _api_Solver:


Solver
******

BYOS
====

The numerical linear algebra solver that you use will ultimately be the
bottleneck of your large scale inversion. To be the most flexible, SimPEG
provides wrappers rather than a comprehensive set of solvers (i.e. BYOS).

The interface is as follows::

    A # Where A is a sparse matrix (or linear operator)
    Ainv = Solver(A, **solverOpts) # Create a solver object with key word arguments
    x = Ainv * b # Where b is a numpy array of shape (n,) or (n,*)
    Ainv.clean() # This cleans the memory footprint(if any)

.. note::

    This is somewhat an abuse of notation for solvers as we never actually
    create A inverse. Instead we are creating an object that acts like A
    inverse, whether that be a Krylov subspace solver or an LU decomposition.

To wrap up solvers in scipy.sparse.linalg it takes one line of code::


    Solver   = SolverWrapD(sp.linalg.spsolve, factorize=False)
    SolverLU = SolverWrapD(sp.linalg.splu, factorize=True)
    SolverCG = SolverWrapI(sp.linalg.cg)

.. note::

    The above solvers are loaded into the base name space of SimPEG.

.. seealso::

    - https://bitbucket.org/petsc/petsc4py
    - https://github.com/bfroehle/pymumps
    - https://github.com/rowanc1/pymatsolver


The API
=======

.. autofunction:: SimPEG.utils.solver_utils.SolverWrapD
    :noindex:

.. autofunction:: SimPEG.utils.solver_utils.SolverWrapI
    :noindex:
