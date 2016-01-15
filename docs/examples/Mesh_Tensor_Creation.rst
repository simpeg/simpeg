.. _examples_Mesh_Tensor_Creation:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..



Mesh: Tensor: Creation
======================

For tensor meshes, there are some functions that can come
in handy. For example, creating mesh tensors can be a bit time
consuming, these can be created speedily by just giving numbers
and sizes of padding. See the example below, that follows this
notation::

    h1 = (
           (cellSize, numPad, [, increaseFactor]),
           (cellSize, numCore),
           (cellSize, numPad, [, increaseFactor])
         )

.. note::

    You can center your mesh by passing a 'C' for the x0[i] position.
    A 'N' will make the entire mesh negative, and a '0' (or a 0) will
    make the mesh start at zero.



.. plot::

    from SimPEG import Examples
    Examples.Mesh_Tensor_Creation.run()

.. literalinclude:: ../../SimPEG/Examples/Mesh_Tensor_Creation.py
    :language: python
    :linenos:
