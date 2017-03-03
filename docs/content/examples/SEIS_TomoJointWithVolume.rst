.. _examples_SEIS_TomoJointWithVolume:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Straight Ray with Volume Data Misfit Term
=========================================

Based on the SEG abstract Heagy, Cockett and Oldenburg, 2014.

Heagy, L. J., Cockett, A. R., & Oldenburg, D. W. (2014, August 5).
Parametrized Inversion Framework for Proppant Volume in a Hydraulically
Fractured Reservoir. SEG Technical Program Expanded Abstracts 2014.
Society of Exploration Geophysicists. doi:10.1190/segam2014-1639.1

This example is a simple joint inversion that consists of a

    - data misfit for the tomography problem
    - data misfit for the volume of the inclusions
      (uses the effective medium theory mapping)
    - model regularization



.. plot::

    from SimPEG import Examples
    Examples.SEIS_TomoJointWithVolume.run()

.. literalinclude:: ../../../SimPEG/Examples/SEIS_TomoJointWithVolume.py
    :language: python
    :linenos:
