.. _examples_PlotData2D:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..



Plotting 2D data
================

Often measured data is in 2D, but locations are not gridded.
Data can be vectoral, hence we want to plot direction and
amplitude of the vector. Following example use SimPEG's
analytic function (electric dipole) to generate data
at 2D plane.



.. plot::

    from SimPEG import Examples
    Examples.PlotData2D.run()

.. literalinclude:: ../../../SimPEG/Examples/PlotData2D.py
    :language: python
    :linenos:
