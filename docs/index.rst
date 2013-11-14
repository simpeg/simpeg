SimPEG
======

SimPEG (Simulation and Parameter Estimation in Geophysics) is a python package for simulation and gradient based parameter estimation in the context of geophysical applications.

The vision is to create a package for finite volume simulation with applications to geophysical imaging and subsurface flow. To enable the understanding of the many different components, this package has the following features:

* modular with respect to the spacial discretization
* is built with the inverse problem in mind
* supports different hyperbolic solvers (Euler, Semi-Lagrangian, Lagrangian)
* supports 2D and 3D problems


Meshing & Operators
===================

.. toctree::
   :maxdepth: 2

   api_BaseMesh
   api_TensorMesh
   api_LogicallyOrthogonalMesh
   api_Cyl1DMesh
   api_DiffOperators
   api_InnerProducts

Forward Problems
================

.. toctree::
   :maxdepth: 2

   api_Problem

Inversion
=========

.. toctree::
   :maxdepth: 2

   api_Optimize

Testing SimPEG
==============

.. toctree::
   :maxdepth: 2

   api_Tests
   api_TestResults


Utility Codes
=============

.. toctree::
   :maxdepth: 2

   api_Utils


Project Index & Search
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

