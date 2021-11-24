import scipy as sp
import numpy as np
from .sparse import Sparse, SparseSmall, SparseDeriv
from .tikhonov import Simple
from .. import utils
from discretize import TensorMesh


class LaterallyConstrained(Sparse):

    def get_grad_horizontal(
        self, xy, hz, dim=3,
        use_cell_weights=True,
        minimum_distance=None
    ):
        """
            Compute Gradient in horizontal direction using Delaunay

        """
        if use_cell_weights:
            self.cell_weights = np.tile(hz, (xy.shape[0], 1)).flatten()

        if dim == 3:
            tri = sp.spatial.Delaunay(xy)
            # Split the triangulation into connections
            edges = np.r_[
                tri.simplices[:, :2],
                tri.simplices[:, 1:],
                tri.simplices[:, [0, 2]]
            ]

            # Sort and keep uniques
            edges = np.sort(edges, axis=1)
            edges = np.unique(
                edges[np.argsort(edges[:, 0]), :], axis=0
            )
            # Compute distance
            if minimum_distance is not None:
                dx = xy[edges[:, 0], 0]-xy[edges[:, 1], 0]
                dy = xy[edges[:, 0], 1]-xy[edges[:, 1], 1]
                distance = np.sqrt(dx**2+dy**2)
                inds = distance < minimum_distance
                edges = edges[inds, :]

            # Create 2D operator, dimensionless for now
            nN = edges.shape[0]
            nStn = xy.shape[0]
            stn, count = np.unique(edges[:, 0], return_counts=True)

            col = []
            row = []
            dm = []
            avg = []
            for ii in range(nN):
                row += [ii]*2
                col += [edges[ii, 0], edges[ii, 1]]
                scale = count[stn == edges[ii, 0]][0]
                dm += [-1., 1.]
                avg += [0.5, 0.5]

            D = sp.sparse.csr_matrix((dm, (row, col)), shape=(nN, nStn))
            A = sp.sparse.csr_matrix((avg, (row, col)), shape=(nN, nStn))

            # Kron vertically for nCz
            Grad = sp.sparse.kron(D, utils.speye(hz.size))
            Avg = sp.sparse.kron(A, utils.speye(hz.size))

            # Override the gradient operator in y-drection
            # This is because of ordering ... See def get_2d_mesh
            # y first then x
            self.regmesh._cellDiffyStencil = self.regmesh.cellDiffxStencil.copy()
            # Override the gradient operator in x-drection
            self.regmesh._cellDiffxStencil = Grad
            # Do the same for the averaging operator
            self.regmesh._aveCC2Fy = self.regmesh.aveCC2Fx.copy()
            self.regmesh._aveCC2Fx = Avg
            self.regmesh._aveFy2CC = self.regmesh.aveFx2CC.copy()
            self.regmesh._aveFx2CC = Avg.T
            return tri

        elif dim == 2:
            # Override the gradient operator in y-drection
            # This is because of ordering ... See def get_2d_mesh
            # y first then x
            temp_x = self.regmesh.cellDiffxStencil.copy()
            temp_y = self.regmesh.cellDiffyStencil.copy()
            self.regmesh._cellDiffyStencil = temp_x
            # Override the gradient operator in x-drection
            self.regmesh._cellDiffxStencil = temp_y
            # Do the same for the averaging operator
            temp_x = self.regmesh.aveCC2Fx.copy()
            temp_y = self.regmesh.aveCC2Fy.copy()
            self.regmesh._aveCC2Fy = temp_x
            self.regmesh._aveCC2Fx = temp_y
            temp_x = self.regmesh.aveCC2Fx.copy()
            temp_y = self.regmesh.aveCC2Fy.copy()
            self.regmesh._aveFy2CC = temp_x
            self.regmesh._aveFx2CC = temp_y
            return True
