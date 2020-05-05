.. _api_Maps:


Maps
****

That's not a map...?!
=====================

A SimPEG map operates on a vector and transforms it to another space.
We will use an example commonly applied in electromagnetics (EM) of the
log-conductivity model.

.. math::

    m = \log(\sigma)

Here we require a *mapping* to get from \\\(m\\\) to  \\\(\\sigma\\\),
we will call this map \\\(\\mathcal{M}\\\).

.. math::

    \sigma = \mathcal{M}(m) = \exp(m)

In SimPEG, we use a (:class:`SimPEG.maps.ExpMap`) to describe how to map
back to conductivity. This is a relatively trivial example (we are just taking
the exponential!) but by defining maps we can start to combine and manipulate
exactly what we think about as our model, \\\(m\\\). In code, this looks like

.. code-block:: python
    :linenos:

    M = Mesh.TensorMesh([100]) # Create a mesh
    expMap = Maps.ExpMap(M)    # Create a mapping
    m = np.zeros(M.nC)         # Create a model vector
    m[M.vectorCCx>0.5] = 1.0   # Set half of it to 1.0
    sig = expMap * m           # Apply the mapping using *
    print(m)
    # [ 0.    0.    0.    1.     1.     1. ]
    print(sig)
    # [ 1.    1.    1.  2.718  2.718  2.718]

Combining Maps
==============

We will use an example where we want a 1D layered earth as
our model, but we want to map this to a 2D discretization to do our forward
modeling. We will also assume that we are working in log conductivity still,
so after the transformation we want to map to conductivity space.
To do this we will introduce the vertical 1D map (:class:`SimPEG.maps.SurjectVertical1D`),
which does the first part of what we just described. The second part will be
done by the :class:`SimPEG.maps.ExpMap` described above.

.. code-block:: python
    :linenos:

    M = Mesh.TensorMesh([7,5])
    v1dMap = Maps.SurjectVertical1D(M)
    expMap = Maps.ExpMap(M)
    myMap = expMap * v1dMap
    m = np.r_[0.2,1,0.1,2,2.9] # only 5 model parameters!
    sig = myMap * m


.. image:: /content/examples/03-maps/images/sphx_glr_plot_combo_001.png
    :target: /content/examples/03-maps/plot_combo.html
    :align: center

If you noticed, it was pretty easy to combine maps. What is even cooler is
that the derivatives also are made for you (if everything goes right).
Just to be sure that the derivative is correct, you should always run the test
on the mapping that you create.

Taking Derivatives
==================

Now that we have wrapped up the mapping, we can ensure that it is easy to take
derivatives (or at least have access to them!). In the :class:`SimPEG.maps.ExpMap`
there are no dependencies between model parameters, so it will be a diagonal matrix:

.. math::

    \left(\frac{\partial \mathcal{M}(m)}{\partial m}\right)_{ii} = \frac{\partial \exp(m_i)}{\partial m} = \exp(m_i)

Or equivalently:

.. math::

    \frac{\partial \mathcal{M}(m)}{\partial m} = \text{diag}( \exp(m) )

The mapping API makes this really easy to test that you have got the derivative correct.
When these are used in the inverse problem, this is extremely important!!

.. plot::
    :include-source:

    import numpy as np
    import discretize
    from SimPEG import maps
    import matplotlib.pyplot as plt
    M = discretize.TensorMesh([100])
    expMap = maps.ExpMap(M)
    m = np.zeros(M.nC)
    m[M.vectorCCx>0.5] = 1.0
    expMap.test(m, plotIt=True)



Common Maps
===========

Exponential Map
---------------

Electrical conductivity varies over many orders of magnitude, so it is a common
technique when solving the inverse problem to parameterize and optimize in terms
of log conductivity. This makes sense not only because it ensures all conductivities
will be positive, but because this is fundamentally the space where conductivity
lives (i.e. it varies logarithmically).

.. autoclass:: SimPEG.maps.ExpMap
    :members:
    :undoc-members:
    :noindex:


Vertical 1D Map
---------------

.. autoclass:: SimPEG.maps.SurjectVertical1D
    :members:
    :undoc-members:
    :noindex:


Map 2D Cross-Section to 3D Model
--------------------------------

.. autoclass:: SimPEG.maps.Surject2Dto3D
    :members:
    :undoc-members:
    :noindex:


Mesh to Mesh Map
----------------

.. autoclass:: SimPEG.maps.Mesh2Mesh
    :members:
    :undoc-members:
    :noindex:


.. image:: /content/examples/03-maps/images/sphx_glr_plot_mesh2mesh_001.png
    :target: /content/examples/03-maps/plot_mesh2mesh.html
    :align: center


Under the Hood
==============

Combo Map
---------

The ComboMap holds the information for multiplying and combining
maps. It also uses the chain rule to create the derivative.
Remember, any time that you make your own combination of mappings
be sure to test that the derivative is correct.

.. autoclass:: SimPEG.maps.ComboMap
    :members:
    :undoc-members:
    :noindex:


The API
=======

The :code:`IdentityMap` is the base class for all mappings, and it does absolutely nothing.

.. automodule:: SimPEG.maps
    :show-inheritance:
    :members:
    :undoc-members:
