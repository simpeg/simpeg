.. _examples_Utils_plot2Ddata:

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
    Examples.Utils_plot2Ddata.run()

.. literalinclude:: ../../../SimPEG/Examples/Utils_plot2Ddata.py
    :language: python
    :linenos:
