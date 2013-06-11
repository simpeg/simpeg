import numpy as np


class Mesh(object):
    """docstring for Mesh"""
    def __init__(self, h):

        if type(h) != list:
            raise Exception("Type of h must be a list variable. e.g. [5, 4, 2] or [[1,1,1],[0.5,0.5]]")

        if np.sum([np.size(x) for x in h]) == len(h):
            # We have specified a shorthand for the mesh e.g. [5, 4, 2]
            # We will recreate the h, such that it lies on the unit cube/square/line
            domain = 1.  # (must be a float)
            h = [np.ones(x)*(domain/x) for x in h]

        dim = len(h)

        if dim > 1 and np.all([len(np.shape(x)) > 1 and np.shape(x)[1] > 1 for x in h]):
            # The h has internal structure, and is not a vector
            # Thus, we must be describing the verticies of the mesh
            # Hence, the mesh is a Logically Orthogonal Mesh
            self.meshType = 'LOM'
        else:
            # Could add other checks, but here the default is a rectangular mesh
            self.meshType = 'RECT'

        if self.meshType != 'LOM':
            # Ensure that the h is a numpy array, with shape: (n,)
            h = [np.array(x).ravel() for x in h]

        # Define the number of nodes
        if self.meshType == 'LOM':
            self._nnodes = np.array(np.shape(h[0]))
        else:
            self._nnodes = np.array([len(x) for x in h]) + 1

        self._nc = self._nnodes - 1
        self._ncells = np.prod(self._nc)
        self._h = h
        self._dim = dim

        m = self._nnodes
        if dim == 1:
            self._nfaces = np.prod(m)
            self._nedges = np.prod(m)
        elif dim == 2:
            self._nfx = m - [0, 1]
            self._nfy = m - [1, 0]
            self._nex = m - [1, 0]
            self._ney = m - [0, 1]

            self._nfaces = [np.prod(self.nfx), np.prod(self.nfy)]
            self._nedges = [np.prod(self.nex), np.prod(self.ney)]
        elif dim == 3:
            self._nfx = m - [0, 1, 1]
            self._nfy = m - [1, 0, 1]
            self._nfz = m - [1, 1, 0]
            self._nex = m - [1, 0, 0]
            self._ney = m - [0, 1, 0]
            self._nez = m - [0, 0, 1]

            self._nfaces = [np.prod(self.nfx), np.prod(self.nfy), np.prod(self.nfz)]
            self._nedges = [np.prod(self.nex), np.prod(self.ney), np.prod(self.nez)]

    def dim():
        doc = "The dimension of the mesh: 1, 2, or 3"
        fget = lambda self: self._dim
        return locals()
    dim = property(**dim())

    def nc():
        doc = "Number of cells in each direction of the mesh"
        fget = lambda self: self._nc
        return locals()
    nc = property(**nc())

    def ncells():
        doc = "Number of cells in the mesh"
        fget = lambda self: self._ncells
        return locals()
    ncells = property(**ncells())

    def nfaces():
        doc = "Number of faces in each direction of the mesh"
        fget = lambda self: self._nfaces
        return locals()
    nfaces = property(**nfaces())

    def nedges():
        doc = "Number of edges in each direction of the mesh"
        fget = lambda self: self._nedges
        return locals()
    nedges = property(**nedges())

    def nfx():
        doc = "Number of faces in the x direction of the mesh"
        fget = lambda self: self._nfx if self.dim > 1 else None
        return locals()
    nfx = property(**nfx())

    def nfy():
        doc = "Number of faces in the y direction of the mesh"
        fget = lambda self: self._nfy if self.dim > 1 else None
        return locals()
    nfy = property(**nfy())

    def nfz():
        doc = "Number of faces in the z direction of the mesh"
        fget = lambda self: self._nfz if self.dim > 2 else None
        return locals()
    nfz = property(**nfz())

    def nex():
        doc = "Number of edges in the x direction of the mesh"
        fget = lambda self: self._nex if self.dim > 1 else None
        return locals()
    nex = property(**nex())

    def ney():
        doc = "Number of edges in the y direction of the mesh"
        fget = lambda self: self._ney if self.dim > 1 else None
        return locals()
    ney = property(**ney())

    def nez():
        doc = "Number of edges in the z direction of the mesh"
        fget = lambda self: self._nez if self.dim > 2 else None
        return locals()
    nez = property(**nez())
