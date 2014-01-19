.. image:: simpeg-logo.png
   :width: 300 px
   :alt: SimPEG
   :align: center

SimPEG (Simulation and Parameter Estimation in Geoscience) is a python
package for simulation and gradient based parameter estimation in the
context of geoscience applications.

The vision is to create a package for finite volume simulation and parameter estimation with
applications to geophysical imaging and subsurface flow. To enable
these goals, this package has the following features:

* is modular with respect to discretization, physics, optimization, and regularization
* is built with the (large-scale) inverse problem in mind
* provides a framework for geophysical and hydrogeologic problems
* supports 1D, 2D and 3D problems


.. raw:: html

   <iframe src="http://row1.ca/labs/simpegvis" width="100%" height="500px" style="border:none;background:#eee;margin:50px 0;"></iframe>


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

* Master Branch
   .. image:: https://travis-ci.org/simpeg/simpeg.png?branch=master
      :target: https://travis-ci.org/simpeg/simpeg
      :alt: Master Branch
      :align: center

* Develop Branch
   .. image:: https://travis-ci.org/simpeg/simpeg.png?branch=develop
      :target: https://travis-ci.org/simpeg/simpeg
      :alt: Develop Branch
      :align: center


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

