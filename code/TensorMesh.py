import numpy as np
from BaseMesh import BaseMesh
from TensorGrid import TensorGrid
from TensorView import TensorView
from utils import ndgrid


class TensorMesh(BaseMesh, TensorGrid, TensorView):
    """
    TensorMesh is a mesh class that deals with tensor product meshes.

    Any Mesh that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    e.g.

        hx = np.array([1,1,1])
        hy = np.array([1,2])
        hz = np.array([1,1,1,1])

        mesh = TensorMesh([hx, hy, hz])

    """
    def __init__(self, h, x0=None):
        super(TensorMesh, self).__init__(np.array([len(x) for x in h]), x0)

        assert len(h) == len(x0), "Dimension mismatch. x0 != len(h)"

        for i, h_i in enumerate(h):
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)

        # Ensure h contains 1D vectors
        self._h = [x.ravel() for x in h]

    def h():
        doc = "h is a list containing the cell widths of the tensor mesh in each dimension."
        fget = lambda self: self._h
        return locals()
    h = property(**h())

    def hx():
        doc = "Width of cells in the x direction"
        fget = lambda self: self._h[0]
        return locals()
    hx = property(**hx())

    def hy():
        doc = "Width of cells in the y direction"
        fget = lambda self: None if self.dim < 2 else self._h[1]
        return locals()
    hy = property(**hy())

    def hz():
        doc = "Width of cells in the z direction"
        fget = lambda self: None if self.dim < 3 else self._h[2]
        return locals()
    hz = property(**hz())

    def vectorNx():
        doc = "Nodal grid vector (1D) in the x direction."
        fget = lambda self: np.r_[0., self.hx.cumsum()] + self.x0[0]
        return locals()
    vectorNx = property(**vectorNx())

    def vectorNy():
        doc = "Nodal grid vector (1D) in the y direction."
        fget = lambda self: None if self.dim < 2 else np.r_[0., self.hy.cumsum()] + self.x0[1]
        return locals()
    vectorNy = property(**vectorNy())

    def vectorNz():
        doc = "Nodal grid vector (1D) in the z direction."
        fget = lambda self: None if self.dim < 3 else np.r_[0., self.hz.cumsum()] + self.x0[2]
        return locals()
    vectorNz = property(**vectorNz())

    def vectorCCx():
        doc = "Cell-centered grid vector (1D) in the x direction."
        fget = lambda self: np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5 + self.x0[0]
        return locals()
    vectorCCx = property(**vectorCCx())

    def vectorCCy():
        doc = "Cell-centered grid vector (1D) in the y direction."
        fget = lambda self: None if self.dim < 2 else np.r_[0, self.hy[:-1].cumsum()] + self.hy*0.5 + self.x0[1]
        return locals()
    vectorCCy = property(**vectorCCy())

    def vectorCCz():
        doc = "Cell-centered grid vector (1D) in the z direction."
        fget = lambda self: None if self.dim < 3 else np.r_[0, self.hz[:-1].cumsum()] + self.hz*0.5 + self.x0[2]
        return locals()
    vectorCCz = property(**vectorCCz())

    def gridCC():
        doc = "Cell-centered grid."

        def fget(self):
            if self._gridCC is None:
                self._gridCC = ndgrid([x for x in [self.vectorCCx, self.vectorCCy, self.vectorCCz] if not x is None])
            return self._gridCC
        return locals()
    _gridCC = None  # Store grid by default
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid."

        def fget(self):
            if self._gridN is None:
                self._gridN = ndgrid([x for x in [self.vectorNx, self.vectorNy, self.vectorNz] if not x is None])
            return self._gridN
        return locals()
    _gridN = None  # Store grid by default
    gridN = property(**gridN())

    def gridFx():
        doc = "Face staggered grid in the x direction."

        def fget(self):
            if self._gridFx is None:
                self._gridFx = ndgrid([x for x in [self.vectorNx, self.vectorCCy, self.vectorCCz] if not x is None])
            return self._gridFx
        return locals()
    _gridFx = None  # Store grid by default
    gridFx = property(**gridFx())

    def gridFy():
        doc = "Face staggered grid in the y direction."

        def fget(self):
            if self._gridFy is None:
                self._gridFy = ndgrid([x for x in [self.vectorCCx, self.vectorNy, self.vectorCCz] if not x is None])
            return self._gridFy
        return locals()
    _gridFy = None  # Store grid by default
    gridFy = property(**gridFy())

    def gridFz():
        doc = "Face staggered grid in the z direction."

        def fget(self):
            if self._gridFz is None:
                self._gridFz = ndgrid([x for x in [self.vectorCCx, self.vectorCCy, self.vectorNz] if not x is None])
            return self._gridFz
        return locals()
    _gridFz = None  # Store grid by default
    gridFz = property(**gridFz())

    def gridEx():
        doc = "Edge staggered grid in the x direction."

        def fget(self):
            if self._gridEx is None:
                self._gridEx = ndgrid([x for x in [self.vectorCCx, self.vectorNy, self.vectorNz] if not x is None])
            return self._gridEx
        return locals()
    _gridEx = None  # Store grid by default
    gridEx = property(**gridEx())

    def gridEy():
        doc = "Edge staggered grid in the y direction."

        def fget(self):
            if self._gridEy is None:
                self._gridEy = ndgrid([x for x in [self.vectorNx, self.vectorCCy, self.vectorNz] if not x is None])
            return self._gridEy
        return locals()
    _gridEy = None  # Store grid by default
    gridEy = property(**gridEy())

    def gridEz():
        doc = "Edge staggered grid in the z direction."

        def fget(self):
            if self._gridEz is None:
                self._gridEz = ndgrid([x for x in [self.vectorNx, self.vectorNy, self.vectorCCz] if not x is None])
            return self._gridEz
        return locals()
    _gridEz = None  # Store grid by default
    gridEz = property(**gridEz())

    def getBoundaryIndex(self, gridType):
        """Needed for faces edges and cells"""
        pass

    def getCellNumbering(self):
        pass


if __name__ == '__main__':
    print('Welcome to tensor mesh!')

    testDim = 1
    h1 = 0.3*np.ones((1, 7))
    h1[:, 0] = 0.5
    h1[:, -1] = 0.6
    h2 = .5 * np.ones((1, 4))
    h3 = .4 * np.ones((1, 6))
    x0 = np.zeros((3, 1))

    if testDim == 1:
        h = [h1]
        x0 = x0[0]
    elif testDim == 2:
        h = [h1, h2]
        x0 = x0[0:2]
    else:
        h = [h1, h2, h3]

    I = np.linspace(0, 1, 8)
    M = TensorMesh(h, x0)

    xn = M.plotGrid()
