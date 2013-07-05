import numpy as np


class BaseMesh(object):
    """BaseMesh does all the counting you don't want to do."""
    def __init__(self, n, x0=None):

        # Check inputs
        if x0 is None:
            x0 = np.zeros(len(n))

        if not len(n) == len(x0):
            raise Exception("Dimension mismatch. x0 != len(n)")

        if len(n) > 3:
            raise Exception("Dimensions higher than 3 are not supported.")

        # Ensure x0 & n are 1D vectors
        self._n = np.array(n, dtype=int).ravel()
        self._x0 = np.array(x0).ravel()
        self._dim = len(n)

    def x0():
        doc = "Origin of the mesh"
        fget = lambda self: self._x0
        return locals()
    x0 = property(**x0())

    def n():
        doc = "Number of Cells in each dimension (array of integers)"
        fget = lambda self: self._n
        return locals()
    n = property(**n())

    def dim():
        doc = "The dimension of the mesh (1, 2, or 3)."
        fget = lambda self: self._dim
        return locals()
    dim = property(**dim())

    def nCx():
        doc = "Number oc cells in the x direction"
        fget = lambda self: self.n[0]
        return locals()
    nCx = property(**nCx())

    def nCy():
        doc = "Number of cells in the y direction"

        def fget(self):
            if self.dim > 1:
                return self.n[1]
            else:
                return None
        return locals()
    nCy = property(**nCy())

    def nCz():
        doc = "Number of cells in the z direction"

        def fget(self):
            if self.dim > 2:
                return self.n[2]
            else:
                return None
        return locals()
    nCz = property(**nCz())

    def nC():
        doc = "Total number of cells"
        fget = lambda self: np.prod(self.n)
        return locals()
    nC = property(**nC())

    def nNx():
        doc = "Number of nodes in the x-direction"
        fget = lambda self: self.nCx + 1
        return locals()
    nNx = property(**nNx())

    def nNy():
        doc = "Number of noes in the y-direction"

        def fget(self):
            if self.dim > 1:
                return self.n[1] + 1
            else:
                return None
        return locals()
    nNy = property(**nNy())

    def nNz():
        doc = "Number of nodes in the z-direction"

        def fget(self):
            if self.dim > 2:
                return self.n[2] + 1
            else:
                return None
        return locals()
    nNz = property(**nNz())

    def nN():
        doc = "Total number of nodes"
        fget = lambda self: self.n + 1
        return locals()
    nN = property(**nN())

    def nEx():
        doc = "Number of x-edges"
        fget = lambda self: np.array([x for x in [self.nCx, self.nNy, self.nNz] if not x is None])
        return locals()
    nEx = property(**nEx())

    def nEy():
        doc = "Number of y-edges"

        def fget(self):
            if self.dim > 1:
                return np.array([x for x in [self.nNx, self.nCy, self.nNz] if not x is None])
            else:
                return None
        return locals()
    nEy = property(**nEy())

    def nEz():
        doc = "Number of z-edges"

        def fget(self):
            if self.dim > 2:
                return np.array([x for x in [self.nNx, self.nNy, self.nCz] if not x is None])
            else:
                return None
        return locals()
    nEz = property(**nEz())

    def nE():
        doc = "Total number of edges"
        fget = lambda self: np.array([np.prod(x) for x in [self.nEx, self.nEy, self.nEz] if not x is None])
        return locals()
    nE = property(**nE())

    def nFx():
        doc = "Number of x-faces"
        fget = lambda self: np.array([x for x in [self.nNx, self.nCy, self.nCz] if not x is None])
        return locals()
    nFx = property(**nFx())

    def nFy():
        doc = "Number of y-faces"

        def fget(self):
            if self.dim > 1:
                return np.array([x for x in [self.nCx, self.nNy, self.nCz] if not x is None])
            else:
                return None
        return locals()
    nFy = property(**nFy())

    def nFz():
        doc = "Number of z-faces"

        def fget(self):
            if self.dim > 2:
                return np.array([x for x in [self.nCx, self.nCy, self.nNz] if not x is None])
            else:
                return None
        return locals()
    nFz = property(**nFz())

    def nF():
        doc = "Total number of faces in each dimension"
        fget = lambda self: np.array([np.prod(x) for x in [self.nFx, self.nFy, self.nFz] if not x is None])
        return locals()
    nF = property(**nF())

if __name__ == '__main__':
    m = BaseMesh([3, 2, 4])
    print m.n
