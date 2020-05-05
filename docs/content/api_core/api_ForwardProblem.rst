.. _api_ForwardProblem:


Forward Simulation
******************

Simulation Class
================

The problem is a partial differential equation of the form:

.. math::

    c(m, u) = 0

Here, \\(m\\) is the model and u is the field (or fields).
Given the model, \\(m\\), we can calculate the fields \\(u(m)\\),
however, the data we collect is a subset of the fields,
and can be defined by a linear projection, \\(P\\).

.. math::

    d_\text{pred} = P u(m)

For the inverse problem, we are interested in how changing the model transforms the data,
as such we can take write the Taylor expansion:

.. math::

    Pu(m + hv) = Pu(m) + hP\frac{\partial u(m)}{\partial m} v + \mathcal{O}(h^2 \left\| v \right\| )

We can linearize and define the sensitivity matrix as:

.. math::

    J = P\frac{\partial u}{\partial m}

The sensitivity matrix, and it's transpose will be used in the inverse problem
to (locally) find how model parameters change the data, and optimize!


Working with the general PDE, \\(c(m, u) = 0\\), where m is the model and u is the field,
the sensitivity is defined as:

.. math::

    J = P\frac{\partial u}{\partial m}

We can take the derivative of the PDE:

.. math::

    \nabla_m c(m, u) \partial m + \nabla_u c(m, u) \partial u = 0

If the forward problem is invertible, then we can rearrange for
\\(\\frac{\\partial u}{\\partial m}\\):

.. math::

    J = - P \left( \nabla_u c(m, u) \right)^{-1} \nabla_m c(m, u)

This can often be computed given a vector (i.e. \\(J(v)\\)) rather than
stored, as \\(J\\) is a large dense matrix.



The API
=======

Simulation
----------

.. autoclass:: SimPEG.simulation.BaseSimulation
    :members:
    :undoc-members:

.. autoclass:: SimPEG.simulation.BaseTimeSimulation
    :members:
    :undoc-members:

.. autoclass:: SimPEG.simulation.LinearSimulation
    :members:
    :undoc-members:

.. autoclass:: SimPEG.simulation.TimeStepArray
    :members:
    :undoc-members:


Fields
------

.. autoclass:: SimPEG.fields.Fields
    :members:
    :undoc-members:

.. autoclass:: SimPEG.fields.TimeFields
    :members:
    :undoc-members:

Survey
------

.. autoclass:: SimPEG.survey.BaseSurvey
    :members:
    :undoc-members:

.. autoclass:: SimPEG.survey.BaseSrc
    :members:
    :undoc-members:

.. autoclass:: SimPEG.survey.BaseRx
    :members:
    :undoc-members:

.. autoclass:: SimPEG.survey.BaseTimeRx
    :members:
    :undoc-members:

.. autoclass:: SimPEG.survey.SourceLocationArray
    :members:
    :undoc-members:

.. autoclass:: SimPEG.survey.RxLocationArray
    :members:
    :undoc-members:

Data
----

.. autoclass:: SimPEG.data.Data
    :members:
    :undoc-members:

.. autoclass:: SimPEG.data.SyntheticData
    :members:
    :undoc-members:

.. autoclass:: SimPEG.data.UncertaintyArray
    :members:
    :undoc-members:
