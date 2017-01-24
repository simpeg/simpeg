.. _examples_PF_Mag_Vec_Inversion_Cartesian:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


PF: Magnetics Vector Inversion - Cartesian
==========================================

In this example, we invert for the 3-component magnetization vector
with the Cartesian formulation. The code is used to invert magnetic
data affected by remanent magnetization and makes no induced
assumption. The inverse problem is three times larger than the usual
susceptibility inversion and depends strongly on the regularization.
The algorithm builtds upon the research done at UBC:

Lelievre, G.P., D.W. Oldenburg, 2009, A 3D total magnetization
inversion applicable when significant, complicated remance is present.
Geophysics, 74, no.3: 21-30

The steps are:
1- SETUP: Create a synthetic model and calculate TMI data. This will
simulate the usual magnetic experiment.

2- INVERSION: Invert for the magnetization vector.



.. plot::

    from SimPEG import Examples
    Examples.PF_Mag_Vec_Inversion_Cartesian.run()

.. literalinclude:: ../../../SimPEG/Examples/PF_Mag_Vec_Inversion_Cartesian.py
    :language: python
    :linenos:
