.. _api_Maps:


Maps
****

A SimPEG Map operates on a vector and transforms it to another space.
We will use an example commonly applied in electromagnetics (EM) of the
log-conductivity model.
Electrical conductivity varies over many orders of magnitude, so it is a common
technique when solving the inverse problem to parameterize and optimize in terms
of log conductivity. This makes sense not only because it ensures all conductivities
will be positive, but because this is fundamentally the space where conductivity
lives (i.e. it varies logarithmically). In SimPEG, we use a (:class:`SimPEG.Maps.ExpMap`) to
describe how to map back to conductivity.

The API
=======

.. autoclass:: SimPEG.Maps.IdentityMap
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Maps.NonLinearMap
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Maps.ComboMap
    :members:
    :undoc-members:

Common Maps
===========

.. autoclass:: SimPEG.Maps.ExpMap
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Maps.Vertical1DMap
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Maps.Mesh2Mesh
    :members:
    :undoc-members:
