.. _examples_Mesh_QuadTree_HangingNodes:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Mesh: QuadTree: Hanging Nodes
=============================

You can give the refine method a function, which is evaluated on every cell
of the TreeMesh.

Occasionally it is useful to initially refine to a constant level
(e.g. 3 in this 32x32 mesh). This means the function is first evaluated
on an 8x8 mesh (2^3).



.. plot::

    from SimPEG import Examples
    Examples.Mesh_QuadTree_HangingNodes.run()

.. literalinclude:: ../../SimPEG/Examples/Mesh_QuadTree_HangingNodes.py
    :language: python
    :linenos:
