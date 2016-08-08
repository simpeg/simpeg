.. _examples_Maps_ComboMaps:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..



Maps: ComboMaps
===============

We will use an example where we want a 1D layered earth as
our model, but we want to map this to a 2D discretization to do our forward
modeling. We will also assume that we are working in log conductivity still,
so after the transformation we want to map to conductivity space.
To do this we will introduce the vertical 1D map (:class:`SimPEG.Maps.SurjectVertical1D`),
which does the first part of what we just described. The second part will be
done by the :class:`SimPEG.Maps.ExpMap` described above.

.. code-block:: python
    :linenos:

    M = Mesh.TensorMesh([7,5])
    v1dMap = Maps.SurjectVertical1D(M)
    expMap = Maps.ExpMap(M)
    myMap = expMap * v1dMap
    m = np.r_[0.2,1,0.1,2,2.9] # only 5 model parameters!
    sig = myMap * m

If you noticed, it was pretty easy to combine maps. What is even cooler is
that the derivatives also are made for you (if everything goes right).
Just to be sure that the derivative is correct, you should always run the test
on the mapping that you create.



.. plot::

    from SimPEG import Examples
    Examples.Maps_ComboMaps.run()

.. literalinclude:: ../../../SimPEG/Examples/Maps_ComboMaps.py
    :language: python
    :linenos:
