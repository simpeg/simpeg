.. _examples_FLOW_Richards_Inverse1D:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


FLOW: Richards: 1D: Inversion
=============================

The example shows an inversion of Richards equation in 1D with a
heterogeneous hydraulic conductivity function.

The haverkamp model is used with the same parameters as Celia1990_
the boundary and initial conditions are also the same. The simulation
domain is 40cm deep and is run for an hour with an exponentially
increasing time step that has a maximum of one minute. The general
setup of the experiment is an infiltration front that advances
downward through the model over time.

The model chosen is the saturated hydraulic conductivity inside
the hydraulic conductivity function (using haverkamp). The initial
model is chosen to be the background (1e-3 cm/s). The saturation data
has 2% random Gaussian noise added.

The figure shows the recovered saturated hydraulic conductivity
next to the true model. The other two figures show the saturation
field for the entire simulation for the true and recovered models.

Rowan Cockett - 21/12/2016

.. _Celia1990: http://www.webpages.uidaho.edu/ch/papers/Celia.pdf


.. plot::

    from SimPEG import Examples
    Examples.FLOW_Richards_Inverse1D.run()

.. literalinclude:: ../../../SimPEG/Examples/FLOW_Richards_Inverse1D.py
    :language: python
    :linenos:
