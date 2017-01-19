.. _examples_PF_Gravity_Laguna_del_Maule_Inversion:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


PF: Gravity: Laguna del Maule Bouguer Gravity
=============================================

This notebook illustrates the SimPEG code used to invert Bouguer
gravity data collected at Laguna del Maule volcanic field, Chile.
Refer to Miller et al 2016 EPSL for full details.

We run the inversion in two steps.  Firstly creating a L2 model and
then applying an Lp norm to produce a compact model.
Craig Miller


.. plot::

    from SimPEG import Examples
    Examples.PF_Gravity_Laguna_del_Maule_Inversion.run()

.. literalinclude:: ../../../SimPEG/Examples/PF_Gravity_Laguna_del_Maule_Inversion.py
    :language: python
    :linenos:
