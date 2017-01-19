.. _examples_PF_Mag_Amplitude_Inversion:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


PF: Magnetic Amplitude Inversion
================================

In this example, we invert magnetic field data simulated
from a simple block model affected by remanent magnetization.
The algorithm builtds upon the research done at CSM:

Li, Y., S. E. Shearer, M. M. Haney, and N. Dannemiller, 2010,
Comprehensive approaches to 3D inversion of magnetic data affected by
remanent magnetization:  Geophysics, 75, no. 1, 1-11

The steps are:
1- SETUP: Create a synthetic model and calculate TMI data. This will
simulate the usual magnetic experiment.

2- PROCESSING: Invert for an equivalent source layer to extract
3-component magnetic field data. The components are then used to
calculate amplitude data.

3- INVERSION: Invert for an effective susceptibility model.



.. plot::

    from SimPEG import Examples
    Examples.PF_Mag_Amplitude_Inversion.run()

.. literalinclude:: ../../../SimPEG/Examples/PF_Mag_Amplitude_Inversion.py
    :language: python
    :linenos:
