.. _examples_FLOW_Richards_Forward1D:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


FLOW: Richards: 1D: Forward Simulation
======================================

The example shows simulation of Richards equation in 1D with a
heterogeneous hydraulic conductivity function.

The haverkamp model is used with the same parameters as Celia1990_
the boundary and initial conditions are also the same. The simulation
domain is 40cm deep and is run for an hour with an exponentially
increasing time step that has a maximum of one minute. The general
setup of the experiment is an infiltration front that advances
downward through the model over time.

Figure (a) shows the heterogeneous saturated hydraulic conductivity
parameter and the location of the data collection, which happens every
minute from 30 seconds into the simulation. Note that the simulation
mesh and the data locations are not aligned, and linear interpolation
is used to collect the data. The points are sampled in pressure head
and then transformed to saturation using the haverkamp model for
the water retention curve.

Figure (b) shows the data collected from the simulation. No noise is
added to the data at this time. The various data locations register
the infiltration event through increasing saturation as the front moves
past the receiver. Notice that the slope of the curves are not equal
as the hydraulic conductivity function is heterogeneous.

Figure (c) shows the saturation field over the entire experiment. Here
you can see that the timestep is not constant over time (5 seconds
at the start of the simulation, 60 seconds at the end). You can also
see the effect of the highly conductive layer in the model between
20 and 25 cm depth. The water drains straight through the conductive
unit and piles up on the other side - advancing the fluid front
faster than the other layers.

Rowan Cockett - 21/12/2016

.. _Celia1990: http://www.webpages.uidaho.edu/ch/papers/Celia.pdf


.. plot::

    from SimPEG import Examples
    Examples.FLOW_Richards_Forward1D.run()

.. literalinclude:: ../../../SimPEG/Examples/FLOW_Richards_Forward1D.py
    :language: python
    :linenos:
