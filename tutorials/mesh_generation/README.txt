Mesh Generation
===============

`SimPEG` uses the `discretize` package to provide a numerical grid (or "mesh") on which to solve differential
equations. Within SimPEG, we use three types of meshes:

	- Tensor meshes
	- Cylindrical meshes (a subclass of the tensor mesh class)
	- Tree meshes

Each mesh type has a similar API to make working with
different meshes relatively simple. Within `discretize`, all meshes are
classes that have properties like the number of cells `nC`, and methods,
like `plotGrid`.

To learn how to create meshes, see the tutorials on the `discretize website <http://discretize.simpeg.xyz/en/master/tutorials/mesh_generation/index.html>`__ . Since the `discretize` package is installed as part of `SimPEG`, meshes can be generated in the same way.