.. _api_Maps:


SimPEG Maps
***********

That's not a map...?!
=====================

A SimPEG Map operates on a vector and transforms it to another space.
We will use an example commonly applied in electromagnetics (EM) of the
log-conductivity model.

.. math::

    m = \log(\sigma)

Here we require a *mapping* to get from \\\(m\\\) to  \\\(\\sigma\\\),
we will call this map \\\(\\mathcal{M}\\\).

.. math::

    \sigma = \mathcal{M}(m) = \exp(m)

In SimPEG, we use a (:class:`SimPEG.Maps.ExpMap`) to describe how to map
back to conductivity. This is a relatively trivial example (we are just taking
the exponential!) but by defining maps we can start to combine and manipulate
exactly what we think about as our model, \\\(m\\\). In code, this looks like

::

    M = Mesh.TensorMesh([100]) # Create a mesh
    expMap = Maps.ExpMap(M)    # Create a mapping
    m = np.zeros(M.nC)         # Create a model vector
    m[M.vectorCCx>0.5] = 1.0   # Set half of it to 1.0
    sig = expMap * m           # Apply the mapping using *
    print m
    # [ 0.    0.    0.    1.     1.     1. ]
    print sig
    # [ 1.    1.    1.  2.718  2.718  2.718]

Combining Maps
==============

We will use an example where we want a 1D layered earth as
our model, but we want to map this to a 2D discretization to do our forward
modeling. We will also assume that we are working in log conductivity still,
so after the transformation we want to map to conductivity space.
To do this we will introduce the vertical 1D map (:class:`SimPEG.Maps.Vertical1DMap`),
which does the first part of what we just described. The second part will be
done by the :class:`SimPEG.Maps.ExpMap` described above.

::

    M = Mesh.TensorMesh([7,5])
    v1dMap = Maps.Vertical1DMap(M)
    expMap = Maps.ExpMap(M)
    myMap = expMap * v1dMap
    m = np.r_[0.2,1,0.1,2,2.9] # only 5 model parameters!
    sig = myMap * m

.. plot::

    from SimPEG import *
    import matplotlib.pyplot as plt
    M = Mesh.TensorMesh([7,5])
    v1dMap = Maps.Vertical1DMap(M)
    expMap = Maps.ExpMap(M)
    myMap = expMap * v1dMap
    m = np.r_[0.2,1,0.1,2,2.9] # only 5 model parameters!
    sig = myMap * m
    figs, axs = plt.subplots(1,2)
    axs[0].plot(m, M.vectorCCy, 'b-o')
    axs[0].set_title('Model')
    axs[0].set_ylabel('Depth, y')
    axs[0].set_xlabel('Value, $m_i$')
    axs[0].set_xlim(0,3)
    axs[0].set_ylim(0,1)
    clbar = plt.colorbar(M.plotImage(sig,ax=axs[1],grid=True,gridOpts=dict(color='grey'))[0])
    axs[1].set_title('Physical Property')
    axs[1].set_ylabel('Depth, y')
    clbar.set_label('$\sigma = \exp(\mathbf{P}m)$')
    plt.tight_layout()

If you noticed, it was pretty easy to combine maps. What is even cooler is
that the derivatives also are made for you (if everything goes right).
Just to be sure that the derivative is correct, you should always run the test
on the mapping that you create.

Taking Derivatives
==================

Now that we have wrapped up the mapping, we can ensure that it is easy to take
derivatives (or at least have access to them!). In the :class:`SimPEG.Maps.ExpMap`
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

    from SimPEG import *
    import matplotlib.pyplot as plt
    M = Mesh.TensorMesh([100])
    expMap = Maps.ExpMap(M)
    m = np.zeros(M.nC)
    m[M.vectorCCx>0.5] = 1.0
    expMap.test(m, plotIt=True)



The API
=======

.. autoclass:: SimPEG.Maps.IdentityMap
    :members:
    :undoc-members:


Common Maps
===========


Exponential Map
---------------

Electrical conductivity varies over many orders of magnitude, so it is a common
technique when solving the inverse problem to parameterize and optimize in terms
of log conductivity. This makes sense not only because it ensures all conductivities
will be positive, but because this is fundamentally the space where conductivity
lives (i.e. it varies logarithmically).

.. autoclass:: SimPEG.Maps.ExpMap
    :members:
    :undoc-members:


Vertical 1D Map
---------------

.. autoclass:: SimPEG.Maps.Vertical1DMap
    :members:
    :undoc-members:


Mesh to Mesh Map
----------------

.. autoclass:: SimPEG.Maps.Mesh2Mesh
    :members:
    :undoc-members:


Some Extras
===========

Combo Map
---------

The ComboMap holds the information for multiplying and combining
maps. It also uses the chain rule create the derivative.
Remember, any time that you make your own combination of mappings
be sure to test that the derivative is correct.

.. autoclass:: SimPEG.Maps.ComboMap
    :members:
    :undoc-members:


Non Linear Map
--------------

.. autoclass:: SimPEG.Maps.NonLinearMap
    :members:
    :undoc-members:
