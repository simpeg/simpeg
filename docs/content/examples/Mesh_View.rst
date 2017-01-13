.. _examples_Mesh_View:

.. --------------------------------- ..
..                                   ..
..    THIS FILE IS AUTO GENEREATED   ..
..                                   ..
..    SimPEG/Examples/__init__.py    ..
..                                   ..
.. --------------------------------- ..


Mesh: Plotting with defining range
==================================

When using a large Mesh with the cylindrical code, it is advantageous
to define a :code:`range_x` and :code:`range_y` when plotting with
vectors. In this case, only the region inside of the range is
interpolated. In particular, you often want to ignore padding cells.



.. plot::

    from SimPEG import Examples
    Examples.Mesh_View.run()

.. literalinclude:: ../../../SimPEG/Examples/Mesh_View.py
    :language: python
    :linenos:
