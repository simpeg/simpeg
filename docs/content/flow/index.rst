.. _api_Richards:


Richards Equation
*****************

There are two different forms of Richards equation that differ
on how they deal with the non-linearity in the time-stepping term.

The most fundamental form, referred to as the
'mixed'-form of Richards Equation [Celia et al., 1990]

.. math::

    \frac{\partial \theta(\psi)}{\partial t} - \nabla \cdot k(\psi) \nabla \psi - \frac{\partial k(\psi)}{\partial z} = 0
    \quad \psi \in \Omega

where theta is water content, and psi is pressure head.
This formulation of Richards equation is called the
'mixed'-form because the equation is parameterized in psi
but the time-stepping is in terms of theta.

As noted in [Celia et al., 1990] the 'head'-based form of Richards
equation can be written in the continuous form as:

.. math::

    \frac{\partial \theta}{\partial \psi}\frac{\partial \psi}{\partial t} - \nabla \cdot k(\psi) \nabla \psi - \frac{\partial k(\psi)}{\partial z} = 0
    \quad \psi \in \Omega

However, it can be shown that this does not conserve mass in the discrete formulation.


Here we reproduce the results from Celia et al. (1990):

.. image:: /content/examples/10-flow/images/sphx_glr_plot_richards_celia1990_001.png
    :target: /content/examples/10-flow/plot_richards_celia1990.html
    :align: center


Richards Problem
================

.. automodule:: SimPEG.FLOW.Richards.RichardsProblem
    :show-inheritance:
    :members:
    :undoc-members:

Richards Survey
===============

.. automodule:: SimPEG.FLOW.Richards.RichardsSurvey
    :show-inheritance:
    :members:
    :undoc-members:


Vadose Zone Empirical Relationships
===================================

.. automodule:: SimPEG.FLOW.Richards.Empirical
    :show-inheritance:
    :members:
    :undoc-members:
