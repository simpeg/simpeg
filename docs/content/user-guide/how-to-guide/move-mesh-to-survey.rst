.. _how-to-move-mesh:

============================
Locating mesh on survey area
============================

The :mod:`discretize` package allows us to define 3D meshes that can be used
for running SimPEG's forward and inverse problems.
Mesh dimensions for :class:`discretize.TensorMesh` and
:class:`discretize.TreeMesh` are assumed to be in meters, and by default their
origin (the westmost-southmost-lowest point) is located in the origin of the
coordinate system (the ``(0, 0, 0)``).

When working with real-world data, we want our mesh to be located around the
survey area. We can move our mesh location by by shifting its ``origin``.

For example, suppose we want to invert some magnetic data from Osborne Mine in
Australia that spans in a region between 448353.0 m and 482422.0 m along the
easting, and between 7578158.0 m and 7594834.0 m along the northing
(UTM zone 54).
Let's also assume that the maximum topographic height of the area is 417
m.

We can build a mesh that spans 34 km on the easting, 17 km on the northing, and
5500 m vertically:

.. code:: python

    import discretize

    dx, dy, dz = 200.0, 200.0, 100.0
    nx, ny, nz = 170, 85, 55
    hx, hy, hz = [(dx, nx)], [(dy, ny)], [(dz, nz)]

    mesh = discretize.TensorMesh([hx, hy, hz])
    print(mesh)


.. code::

    TensorMesh: 794,750 cells

                        MESH EXTENT             CELL WIDTH      FACTOR
    dir    nC        min           max         min       max      max
    ---   ---  ---------------------------  ------------------  ------
     x    170          0.00     34,000.00    200.00    200.00    1.00
     y     85          0.00     17,000.00    200.00    200.00    1.00
     z     55          0.00      5,500.00    100.00    100.00    1.00


The ``origin`` of this mesh is located on ``(0, 0, 0)``. We can move it to the
survey area by shifting it to (448353.0 m, 7578158.0 m, -5000 m):

.. code:: python

    mesh.origin = (448353.0, 7578158.0, -5000)
    print(mesh)

.. code::

   TensorMesh: 794,750 cells

                       MESH EXTENT             CELL WIDTH      FACTOR
   dir    nC        min           max         min       max      max
   ---   ---  ---------------------------  ------------------  ------
    x    170    448,353.00    482,353.00    200.00    200.00    1.00
    y     85  7,578,158.00  7,595,158.00    200.00    200.00    1.00
    z     55     -5,000.00        500.00    100.00    100.00    1.00


By shifting the ``origin`` we are not changing the number of cells in the mesh
nor their dimensions. We are just moving the location of the mesh in the three
directions.

.. note::

   We shift the z coordinate of the origin to -5000 m so we leave 500 m above
   the zeroth height to possibly account for topography.


.. tip::

   Alternatively, we can set the ``origin`` when defining the mesh, by passing
   it as an argument. For example:

   .. code:: python

        origin = (448353.0, 7578158.0, -5000)
        mesh = discretize.TensorMesh([hx, hy, hz], origin=origin)
        print(mesh)


Considering padding: simple case
--------------------------------

It's best practice to add some padding to the mesh when using it in an
inversion. The padding cells will allocate any potential body outside the
survey area, which effect might be present in the data.

Let's take the previous example and build a mesh that has 3 km of padding
on each horizontal direction:

.. code:: python

    hx = [(200.0, 15), (dx, nx), (200.0, 15)]
    hy = [(200.0, 15), (dy, ny), (200.0, 15)]
    hz = [(dz, nz)]

    mesh = discretize.TensorMesh([hx, hy, hz])
    print(mesh)

.. code::

    TensorMesh: 1,265,000 cells

                        MESH EXTENT             CELL WIDTH      FACTOR
    dir    nC        min           max         min       max      max
    ---   ---  ---------------------------  ------------------  ------
     x    200          0.00     40,000.00    200.00    200.00    1.00
     y    115          0.00     23,000.00    200.00    200.00    1.00
     z     55          0.00      5,500.00    100.00    100.00    1.00

Now we can shift the ``origin``, but we also need to take into account the
padding cells. We will set the origin to the westmost-southmost corner of the
survey minus the padding distance we added to the mesh (3km):

.. code:: python

    mesh.origin = (448353.0 - 3000, 7578158.0 - 3000, -5000)
    print(mesh)

.. code::

    TensorMesh: 1,265,000 cells

                        MESH EXTENT             CELL WIDTH      FACTOR
    dir    nC        min           max         min       max      max
    ---   ---  ---------------------------  ------------------  ------
     x    200    445,353.00    485,353.00    200.00    200.00    1.00
     y    115  7,575,158.00  7,598,158.00    200.00    200.00    1.00
     z     55     -5,000.00        500.00    100.00    100.00    1.00


Considering padding: padding factor
-----------------------------------

Alternatively, we can introduce padding through a *padding factor*. Instead of
creating padding cells of the same size, we can use the padding factor to
create padding cells that increase in volume as they move away from the survey
area.
This is the usual approach to add padding cells to
:class:`discretize.TensorMesh` since it reduces the amount of cells in the
mesh, making inversions less expensive.

Following the previous example, let's add 7 cells to each side of the
horizontal directions. Let's make the first cells the same size of the ones in
the mesh, and then start increasing their size with a factor of 1.5:

.. code:: python

    n_pad_cells = 7
    factor = 1.5

    hx = [(dx, n_pad_cells, -factor), (dx, nx), (dx, n_pad_cells, factor)]
    hy = [(dy, n_pad_cells, -factor), (dy, ny), (dy, n_pad_cells, factor)]
    hz = [(dz, nz)]

    mesh = discretize.TensorMesh([hx, hy, hz])
    print(mesh)

.. code::

    TensorMesh: 1,001,880 cells

                        MESH EXTENT             CELL WIDTH      FACTOR
    dir    nC        min           max         min       max      max
    ---   ---  ---------------------------  ------------------  ------
     x    184          0.00     53,303.12    200.00  3,417.19    1.50
     y     99          0.00     36,303.12    200.00  3,417.19    1.50
     z     55          0.00      5,500.00    100.00    100.00    1.00


As before, we need to consider the padding cells when shifting the ``origin``
of the mesh. Since we know that we added 7 cells to each side, we can leverage
that by shifting the 7th node of the x and y axes to the westmost-southmost
corner of the survey:

.. code:: python

    x_node_7th = mesh.nodes_x[n_pad_cells]
    y_node_7th = mesh.nodes_y[n_pad_cells]
    mesh.origin = (448353.0 - x_node_7th, 7578158.0 - y_node_7th, -5000)
    print(mesh)

.. code::

    TensorMesh: 1,001,880 cells

                        MESH EXTENT             CELL WIDTH      FACTOR
    dir    nC        min           max         min       max      max
    ---   ---  ---------------------------  ------------------  ------
     x    184    438,701.44    492,004.56    200.00  3,417.19    1.50
     y     99  7,568,506.44  7,604,809.56    200.00  3,417.19    1.50
     z     55     -5,000.00        500.00    100.00    100.00    1.00
