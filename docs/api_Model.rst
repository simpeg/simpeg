.. _api_Model:


Model
*****

A SimPEG model operates on a vector and transforms it to another space.
We will use an example commonly applied in electromagnetics (EM) of the
log-conductivity model (:class:`SimPEG.Model.LogModel`).
Electrical conductivity varies over many orders of magnitude, so it is a common
technique when solving the inverse problem to parameterize and optimize in terms
of log conductivity. This makes sense not only because it ensures all conductivities
will be positive, but because this is fundamentally the space where conductivity
lives (i.e. it varies logarithmically). In SimPEG, we use the term Model to
describe how to get between these two spaces.

The API
=======

.. autoclass:: SimPEG.Model.BaseModel
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Model.BaseNonLinearModel
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Model.ComboModel
    :members:
    :undoc-members:

Common Models
=============

.. autoclass:: SimPEG.Model.LogModel
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Model.Vertical1DModel
    :members:
    :undoc-members:

.. autoclass:: SimPEG.Model.Mesh2Mesh
    :members:
    :undoc-members:
