.. _examples_SEIS_VolumeRegularization:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Straight Ray with Volume Regularization
=======================================

Based on the SEG abstract Heagy, Cockett and Oldenburg, 2014.

This example is a simple joint inversion that consists of a

    - data misfit for the tomography problem
    - data misfit for the volume of the inclusions
      (uses the effective medium theory mapping)
    - model regularization



.. plot::

    from SimPEG import Examples
    Examples.SEIS_VolumeRegularization.run()

.. literalinclude:: ../../../SimPEG/Examples/SEIS_VolumeRegularization.py
    :language: python
    :linenos:
