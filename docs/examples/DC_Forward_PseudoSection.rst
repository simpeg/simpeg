.. _examples_DC_Forward_PseudoSection:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


DC Forward Simulation
=====================

Forward model two conductive spheres in a half-space and plot a
pseudo-section. Assumes an infinite line source and measures along the
center of the spheres.

INPUT:
loc     = Location of spheres [[x1,y1,z1],[x2,y2,z2]]
radi    = Radius of spheres [r1,r2]
param   = Conductivity of background and two spheres [m0,m1,m2]
stype   = survey type "pdp" (pole dipole) or "dpdp" (dipole dipole)
dtype   = Data type "appr" (app res) | "appc" (app cond) | "volt" (potential)
Created by @fourndo



.. plot::

    from SimPEG import Examples
    Examples.DC_Forward_PseudoSection.run()

.. literalinclude:: ../../SimPEG/Examples/DC_Forward_PseudoSection.py
    :language: python
    :linenos:
