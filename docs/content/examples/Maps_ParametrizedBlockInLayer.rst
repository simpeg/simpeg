.. _examples_Maps_ParametrizedBlockInLayer:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Maps: Parametrized Block in a Layer
===================================

Parametrized description of a block confined to a layer in a
wholespace. The mapping can be applied in 2D or 3D. Here we show a 2D
example.

The model is given by

.. code::

    m = np.r_[
       'value of the background',
       'value in the layer',
       'value in the block',
       'center of the layer (depth)',
       'thickness of the layer',
       'x-center of block',
       'width of the block'
    ]



.. plot::

    from SimPEG import Examples
    Examples.Maps_ParametrizedBlockInLayer.run()

.. literalinclude:: ../../../SimPEG/Examples/Maps_ParametrizedBlockInLayer.py
    :language: python
    :linenos:
