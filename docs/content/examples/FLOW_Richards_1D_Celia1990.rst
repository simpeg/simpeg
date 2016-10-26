.. _examples_FLOW_Richards_1D_Celia1990:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


FLOW: Richards: 1D: Celia1990
=============================

There are two different forms of Richards equation that differ
on how they deal with the non-linearity in the time-stepping term.

The most fundamental form, referred to as the
'mixed'-form of Richards Equation Celia1990_

.. math::

    \frac{\partial \theta(\psi)}{\partial t} -
    \nabla \cdot k(\psi) \nabla \psi -
    \frac{\partial k(\psi)}{\partial z} = 0
    \quad \psi \in \Omega

where \\(\\theta\\) is water content, and \\(\\psi\\)
is pressure head. This formulation of Richards equation is called the
'mixed'-form because the equation is parameterized in \\(\\psi\\)
but the time-stepping is in terms of \\(\\theta\\).

As noted in Celia1990_ the 'head'-based form of Richards
equation can be written in the continuous form as:

.. math::

    \frac{\partial \theta}{\partial \psi}
    \frac{\partial \psi}{\partial t} -
    \nabla \cdot k(\psi) \nabla \psi -
    \frac{\partial k(\psi)}{\partial z} = 0
    \quad \psi \in \Omega

However, it can be shown that this does not conserve mass in the
discrete formulation.

Here we reproduce the results from Celia1990_ demonstrating the
head-based formulation and the mixed-formulation.

.. _Celia1990: http://www.webpages.uidaho.edu/ch/papers/Celia.pdf


.. plot::

    from SimPEG import Examples
    Examples.FLOW_Richards_1D_Celia1990.run()

.. literalinclude:: ../../../SimPEG/Examples/FLOW_Richards_1D_Celia1990.py
    :language: python
    :linenos:
