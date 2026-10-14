.. _pardiso-threads:

Setting the number of Pardiso threads
=====================================

The :class:`pymatsolver.Pardiso` solver can use multiple threads to solve a
linear system. This guide shows how to set its thread count from a SimPEG
simulation. It requires an x86_64 machine with ``pydiso`` installed; see
:ref:`choosing-solvers` for other solver options.

Pass the ``n_threads`` option through the simulation's ``solver_opts``
dictionary. For example, the following DC simulation requests two threads:

.. code:: python

   import numpy as np
   from discretize import TensorMesh
   from pymatsolver import Pardiso
   import simpeg.electromagnetics.static.resistivity as dc

   h = [(1.0, 10)]
   mesh = TensorMesh([h, h, h], origin="CCC")
   receiver = dc.receivers.Dipole(locations_m=(-1, 0, 0), locations_n=(1, 0, 0))
   source = dc.sources.Dipole(
       receiver_list=[receiver], location_a=(-2, 0, 0), location_b=(2, 0, 0)
   )
   survey = dc.Survey([source])

   simulation = dc.Simulation3DNodal(
       mesh=mesh,
       survey=survey,
       sigma=np.full(mesh.n_cells, 1e-2),
       solver=Pardiso,
       solver_opts={"n_threads": 2},
   )
   predicted_data = simulation.dpred()

SimPEG passes ``solver_opts`` to the solver when it creates a solver instance.
In this example, that happens while computing ``predicted_data``, not when
constructing the simulation. Setting ``solver_opts`` does not change a solver
instance that the simulation has already created.

.. important::

   Pardiso's thread setting is shared by all Pardiso solver instances in the
   same Python process. It is not a separate limit for each simulation.
   Creating another Pardiso solver with a different ``n_threads`` value changes
   the setting for existing instances too. Separate Python processes each have
   their own setting.

You can also inspect or change the setting through an existing Pardiso solver's
``n_threads`` property. The following example illustrates its process-wide
effect with two small linear systems:

.. code:: python

   from scipy.sparse import eye

   matrix = eye(3, format="csr")
   first_solver = Pardiso(matrix, n_threads=2)
   second_solver = Pardiso(matrix, n_threads=1)
   assert first_solver.n_threads == second_solver.n_threads == 1

   first_solver.n_threads = 2
   assert first_solver.n_threads == second_solver.n_threads == 2

   first_solver.clean()
   second_solver.clean()

Choose a positive thread count appropriate for the resources available to your
process. More threads do not necessarily make a small problem faster. If you
run simulations in multiple processes, account for the threads used by every
process to avoid requesting more CPU resources than are available.
