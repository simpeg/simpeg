.. _choosing-solvers:

================
Choosing solvers
================

Several simulations available in SimPEG need to numerically solve a partial
differential equations system (PDE), such as
:class:`~simpeg.electromagnetics.static.resistivity.Simulation3DNodal` (DC)
:class:`~simpeg.electromagnetics.time_domain.Simulation3DMagneticFluxDensity`
(TDEM)
and
:class:`~simpeg.electromagnetics.frequency_domain.Simulation3DMagneticFluxDensity`
(FDEM).
A numerical solver is needed to solve the PDEs.
SimPEG can make use of the solvers available in :mod:`pymatsolver`, like
:class:`pymatsolver.Pardiso`, :class:`pymatsolver.Mumps` or
:class:`pymatsolver.SolverLU`.
The choice of an appropriate solver can affect the computation time required to
solve the PDE. Generally we recommend using direct solvers over iterative solvers
for SimPEG, but be aware that direct solvers have much larger memory requirements.

- The :class:`~pymatsolver.Pardiso` solver wraps the `oneMKL PARDISO <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_ solver available for x86_64 CPUs.
- The :class:`~pymatsolver.Mumps` solver wraps `MUMPS <https://mumps-solver.org/index.php?page=home>`_, a fast solver available for all CPU brands, including Apple silicon architecture.
- The :class:`~pymatsolver.SolverLU` wraps SciPy's :func:`scipy.sparse.linalg.splu`. The performance of this solver is not up to the level of :class:`~pymatsolver.Mumps` and :class:`~pymatsolver.Pardiso`. Usage of the :class:`~pymatsolver.SolveLU` is recommended only when it's not possible to use other faster solvers.


The default solver
------------------

We can use :func:`simpeg.utils.get_default_solver` to obtain a reasonable default
solver available for our system:

.. code:: python

   import simpeg
   import simpeg.electromagnetics.static.resistivity as dc

   # Get default solver
   solver = simpeg.utils.get_default_solver()
   print(f"Solver: {solver}")

which would print out on an x86_64 cpu:

.. code::

    Solver: <class 'pymatsolver.direct.pardiso.Pardiso'>

We can then use this solver in a simulation:

.. code:: python

   # Define a simple mesh
   h = [(1.0, 10)]
   mesh = discretize.TensorMesh([h, h, h], origin="CCC")

   # And a DC survey
   receiver = dc.receivers.Dipole(locations_m=(-1, 0, 0), locations_n=(1, 0, 0))
   source = dc.sources.Dipole(
       receiver_list=[receiver], location_a=(-2, 0, 0), location_b=(2, 0, 0)
   )
   survey = dc.Survey([source])

   # Use the default solver in the simulation
   simulation = dc.Simulation3DNodal(mesh=mesh, survey=survey, solver=solver)

.. note::

    The priority list used to choose a default solver is:

    1) :class:`pymatsolver.Pardiso`
    2) :class:`pymatsolver.Mumps`
    3) :class:`pymatsolver.SolverLU`


Setting solvers manually
------------------------

Alternatively, we can manually set a solver. For example, if we want to use
MUMPS in our DC resistivity simulation, we can import
:class:`pymatsolver.Mumps` and pass it to our simulation:

.. code:: python

   import simpeg.electromagnetics.static.resistivity as dc
   from pymatsolver import Mumps

   # Manually set Mumps as our solver
   simulation = dc.Simulation3DNodal(mesh=mesh, survey=survey, solver=Mumps)

.. note::

   When sharing this code with a colleague, keep in mind that
   it might not work if MUMPS is not available in their system.

   For such scenarios, we recommend using the
   :func:`simpeg.utils.get_default_solver` function, that will always return
   a suitable solver for the current system.

Ultimately, choosing the best solver is a mixture of the problem you are solving
and your current system. Experiment with different solvers yourself to choose
the best.

