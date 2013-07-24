import numpy as np
from BaseMesh import BaseMesh
from DiffOperators import DiffOperators
from utils import mkvc, ndgrid


class LogicallyOrthogonalMesh(BaseMesh, DiffOperators):  # , LOMGrid
    """
    LogicallyOrthogonalMesh is a mesh class that deals with logically orthogonal meshes.

    """
    def __init__(self, nodes, x0=None):
        # Start with some error checking:
        assert type(nodes) == list, "'nodes' variable must be a list of np.ndarray"

        for i, nodes_i in enumerate(nodes):
            assert type(nodes_i) == np.ndarray, ("nodes[%i] is not a numpy array." % i)
            assert nodes_i.shape == nodes[0].shape, ("nodes[%i] is not the same shape as nodes[0]" % i)

        assert len(nodes[0].shape) == len(nodes), "Dimension mismatch"

        super(LogicallyOrthogonalMesh, self).__init__(np.array(nodes[0].shape)-1, x0)

        assert len(nodes[0].shape) == len(self.x0), "Dimension mismatch. x0 != len(h)"

        # Save nodes to private variable _gridN as vectors
        self._gridN = np.ones((nodes[0].size, self.dim))
        for i, node_i in enumerate(nodes):
            self._gridN[:, i] = mkvc(node_i)

    def gridCC():
        doc = "Cell-centered grid."

        def fget(self):
            if self._gridCC is None:
                ccV = (self.nodalVectorAve*mkvc(self.gridN))
                self._gridCC = ccV.reshape((-1, self.dim), order='F')
            return self._gridCC
        return locals()
    _gridCC = None  # Store grid by default
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid."

        def fget(self):
            if self._gridN is None:
                raise Exception("Someone deleted this. I blame you.")
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

if __name__ == '__main__':
    nc = 5
    h1 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h2 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h3 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h = [h1, h2, h3]
    X, Y, Z = ndgrid(h1, h2, h3, vector=False)
    M = LogicallyOrthogonalMesh([X, Y, Z])
    print M.gridCC[:,0]
    print M.gridN[:,0]
