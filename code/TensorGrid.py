import numpy as np
from utils import ndgrid


class TensorGrid(object):
    """
    Define nodal, cell-centered and staggered tensor grids for 1, 2 and 3
    dimensions.

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

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
