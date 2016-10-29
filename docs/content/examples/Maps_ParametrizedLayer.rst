.. _examples_Maps_ParametrizedLayer:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Maps: Parametrized Layer
========================

Build a model of a parametrized layer in a wholespace. If you want to
build a model of a parametrized layer in a halfspace, also use
Maps.InjectActiveCell.

The model is

.. code::

    m = [
        'background physical property value',
        'layer physical property value',
        'layer center',
        'layer thickness'
    ]



.. plot::

    from SimPEG import Examples
    Examples.Maps_ParametrizedLayer.run()

.. literalinclude:: ../../../SimPEG/Examples/Maps_ParametrizedLayer.py
    :language: python
    :linenos:
