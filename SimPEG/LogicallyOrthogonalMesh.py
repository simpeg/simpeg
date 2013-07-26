import numpy as np
from BaseMesh import BaseMesh
from DiffOperators import DiffOperators
from utils import mkvc, ndgrid


class LogicallyOrthogonalMesh(BaseMesh, DiffOperators):  # , LOMGrid
    """
    LogicallyOrthogonalMesh is a mesh class that deals with logically orthogonal meshes.

    """
    def __init__(self, nodes, x0=None):
        assert type(nodes) == list, "'nodes' variable must be a list of np.ndarray"

        for i, nodes_i in enumerate(nodes):
            assert type(nodes_i) == np.ndarray, ("nodes[%i] is not a numpy array." % i)
            assert nodes_i.shape == nodes[0].shape, ("nodes[%i] is not the same shape as nodes[0]" % i)

        assert len(nodes[0].shape) == len(nodes), "Dimension mismatch"
        assert len(nodes[0].shape) > 1, "Not worth using LOM for a 1D mesh."

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

    # --------------- Geometries ---------------------
    def vol():
        doc = "Construct cell volumes of the 3D model as 1d array."

        def fget(self):
            if(self._vol is None):
                vh = self.h
                # Compute cell volumes
                if(self.dim == 1):
                    self._vol = mkvc(vh[0])
                elif(self.dim == 2):
                    # Cell sizes in each direction
                    self._vol = mkvc(np.outer(vh[0], vh[1]))
                elif(self.dim == 3):
                    # Cell sizes in each direction
                    self._vol = mkvc(np.outer(mkvc(np.outer(vh[0], vh[1])), vh[2]))
            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())


if __name__ == '__main__':
    nc = 5
    h1 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h2 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h3 = np.cumsum(np.r_[0, np.ones(nc)/(nc)])
    h = [h1, h2, h3]
    X, Y, Z = ndgrid(h1, h2, h3, vector=False)
    M = LogicallyOrthogonalMesh([X, Y, Z])
    print M.r(M.gridCC, format='M')
    print M.gridN[:, 0]
