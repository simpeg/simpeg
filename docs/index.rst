.. SimPEG documentation master file, created by
   sphinx-quickstart on Fri Aug 30 18:42:44 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SimPEG
======

SimPEG (Simulation and Parameter Estimation in Geophysics) is a python package for simulation and gradient based parameter estimation in the context of geophysical applications.

The vision is to create a package for finite volume simulation with applications to geophysical imaging and subsurface flow. To enable the understanding of the many different components, this package has the following features:

* modular with respect to the spacial discretization
* is built with the inverse problem in mind
* supports different hyperbolic solvers (Euler, Semi-Lagrangian, Lagrangian)
* supports 2D and 3D problems


Contents
========

.. toctree::
   :maxdepth: 2

   api_BaseMesh
   api_TensorMesh

.. plot:: examples/mesh/plot_grid.py
   :include-source:


Project Index & Search
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

