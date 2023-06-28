import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .. import props
from .. import utils

###############################################################################
#                                                                             #
#                             Regularization Mesh                             #
#                                                                             #
###############################################################################


class RegularizationMesh(props.BaseSimPEG):
    """
    **Regularization Mesh**

    This contains the operators used in the regularization. Note that these
    are not necessarily true differential operators, but are constructed from
    a `discretize` Mesh.

    :param discretize.base.BaseMesh mesh: problem mesh
    :param numpy.ndarray indActive: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells

    """

    regularization_type = None  # or 'Simple', 'Sparse' or 'Tikhonov'

    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        utils.setKwargs(self, **kwargs)

    indActive = properties.Array("active indices in mesh", dtype=[bool, int])

    @properties.validator("indActive")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

    @property
    def vol(self):
        """
        reduced volume vector

        :rtype: numpy.ndarray
        :return: reduced cell volume
        """
        if getattr(self, "_vol", None) is None:
            self._vol = self.Pac.T * self.mesh.cell_volumes
        return self._vol

    @property
    def nC(self):
        """
        reduced number of cells

        :rtype: int
        :return: number of cells being regularized
        """
        if self.indActive is not None:
            return int(self.indActive.sum())
        return self.mesh.nC

    @property
    def dim(self):
        """
        dimension of regularization mesh (1D, 2D, 3D)

        :rtype: int
        :return: dimension
        """
        if getattr(self, "_dim", None) is None:
            self._dim = self.mesh.dim
        return self._dim

    @property
    def Pac(self):
        """
        projection matrix that takes from the reduced space of active cells to
        full modelling space (ie. nC x nindActive)

        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.indActive is None:
                self._Pac = utils.speye(self.mesh.nC)
            else:
                self._Pac = utils.speye(self.mesh.nC)[:, self.indActive]
        return self._Pac

    @property
    def Pafx(self):
        """
        projection matrix that takes from the reduced space of active x-faces
        to full modelling space (ie. nFx x nindActive_Fx )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, "_Pafx", None) is None:
            # if getattr(self.mesh, 'aveCC2Fx', None) is not None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                ind_active = self.indActive
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                indActive_Fx = (
                    self.mesh.average_cell_to_total_face_x() * ind_active
                ) >= 1
                self._Pafx = utils.speye(self.mesh.ntFx)[:, indActive_Fx]
            else:
                if self.indActive is None:
                    self._Pafx = utils.speye(self.mesh.nFx)
                else:
                    indActive_Fx = self.mesh.aveFx2CC.T * self.indActive >= 1
                    self._Pafx = utils.speye(self.mesh.nFx)[:, indActive_Fx]
        return self._Pafx

    @property
    def Pafy(self):
        """
        projection matrix that takes from the reduced space of active y-faces
        to full modelling space (ie. nFy x nindActive_Fy )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-y projection matrix
        """
        if getattr(self, "_Pafy", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                ind_active = self.indActive
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                indActive_Fy = (
                    self.mesh.average_cell_to_total_face_y() * ind_active
                ) >= 1
                self._Pafy = utils.speye(self.mesh.ntFy)[:, indActive_Fy]
            else:
                if self.indActive is None:
                    self._Pafy = utils.speye(self.mesh.nFy)
                else:
                    indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) >= 1
                    self._Pafy = utils.speye(self.mesh.nFy)[:, indActive_Fy]
        return self._Pafy

    @property
    def Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nindActive_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-z projection matrix
        """
        if getattr(self, "_Pafz", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                ind_active = self.indActive
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                indActive_Fz = (
                    self.mesh.average_cell_to_total_face_z() * ind_active
                ) >= 1
                self._Pafz = utils.speye(self.mesh.ntFz)[:, indActive_Fz]
            else:
                if self.indActive is None:
                    self._Pafz = utils.speye(self.mesh.nFz)
                else:
                    indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) >= 1
                    self._Pafz = utils.speye(self.mesh.nFz)[:, indActive_Fz]
        return self._Pafz

    @property
    def average_face_to_cell(self):
        if self.dim == 1:
            return self.aveFx2CC
        elif self.dim == 2:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC])
        else:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC, self.aveFz2CC])

    @property
    def aveFx2CC(self):
        """
        averaging from active cell centers to active x-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active x-faces
        """
        if getattr(self, "_aveFx2CC", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                nCinRow = utils.mkvc((self.aveCC2Fx.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFx2CC = utils.sdiag(nCinRow) * self.aveCC2Fx.T

            else:
                self._aveFx2CC = self.Pac.T * self.mesh.aveFx2CC * self.Pafx

        return self._aveFx2CC

    @property
    def aveCC2Fx(self):
        """
        averaging from active x-faces to active cell centers

        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active x-faces to active cell centers
        """
        if getattr(self, "_aveCC2Fx", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                self._aveCC2Fx = (
                    self.Pafx.T * self.mesh.average_cell_to_total_face_x() * self.Pac
                )
            else:
                self._aveCC2Fx = (
                    utils.sdiag(1.0 / (self.aveFx2CC.T).sum(1)) * self.aveFx2CC.T
                )
        return self._aveCC2Fx

    @property
    def aveFy2CC(self):
        """
        averaging from active cell centers to active y-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active y-faces
        """
        if getattr(self, "_aveFy2CC", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                nCinRow = utils.mkvc((self.aveCC2Fy.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFy2CC = utils.sdiag(nCinRow) * self.aveCC2Fy.T
            else:
                self._aveFy2CC = self.Pac.T * self.mesh.aveFy2CC * self.Pafy

        return self._aveFy2CC

    @property
    def aveCC2Fy(self):
        """
        averaging from active y-faces to active cell centers

        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active y-faces to active cell centers
        """
        if getattr(self, "_aveCC2Fy", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                self._aveCC2Fy = (
                    self.Pafy.T * self.mesh.average_cell_to_total_face_y() * self.Pac
                )
            else:
                self._aveCC2Fy = (
                    utils.sdiag(1.0 / (self.aveFy2CC.T).sum(1)) * self.aveFy2CC.T
                )
        return self._aveCC2Fy

    @property
    def aveFz2CC(self):
        """
        averaging from active cell centers to active z-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active z-faces
        """
        if getattr(self, "_aveFz2CC", None) is None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                nCinRow = utils.mkvc((self.aveCC2Fz.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFz2CC = utils.sdiag(nCinRow) * self.aveCC2Fz.T
            else:
                self._aveFz2CC = self.Pac.T * self.mesh.aveFz2CC * self.Pafz

        return self._aveFz2CC

    @property
    def aveCC2Fz(self):
        """
        averaging from active z-faces to active cell centers

        :rtype: scipy.sparse.csr_matrix
        :return: averaging matrix from active z-faces to active cell centers
        """
        if getattr(self, "_aveCC2Fz", None) is None:
            # if getattr(self.mesh, 'aveCC2Fz', None) is not None:
            if self.mesh._meshType == "TREE" and self.regularization_type != "Tikhonov":
                self._aveCC2Fz = (
                    self.Pafz.T * self.mesh.average_cell_to_total_face_z() * self.Pac
                )
            else:
                self._aveCC2Fz = (
                    utils.sdiag(1.0 / (self.aveFz2CC.T).sum(1)) * self.aveFz2CC.T
                )
        return self._aveCC2Fz

    @property
    def cell_gradient(self):
        if self.dim == 1:
            return self.cellDiffx
        elif self.dim == 2:
            return sp.vstack([self.cellDiffx, self.cellDiffy])
        else:
            return sp.vstack([self.cellDiffx, self.cellDiffy, self.cellDiffz])

    @property
    def cellDiffx(self):
        """
        cell centered difference in the x-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, "_cellDiffx", None) is None:
            self._cellDiffx = self.Pafx.T * self.mesh.cell_gradient_x * self.Pac
        return self._cellDiffx

    @property
    def cellDiffy(self):
        """
        cell centered difference in the y-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if getattr(self, "_cellDiffy", None) is None:
            self._cellDiffy = self.Pafy.T * self.mesh.cellGrady * self.Pac
        return self._cellDiffy

    @property
    def cellDiffz(self):
        """
        cell centered difference in the z-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the z-direction
        """
        if getattr(self, "_cellDiffz", None) is None:
            self._cellDiffz = self.Pafz.T * self.mesh.cellGradz * self.Pac
        return self._cellDiffz

    @property
    def faceDiffx(self):
        """
        x-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the x-direction
        """
        if getattr(self, "_faceDiffx", None) is None:
            self._faceDiffx = self.Pac.T * self.mesh.face_x_divergence * self.Pafx
        return self._faceDiffx

    @property
    def faceDiffy(self):
        """
        y-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the y-direction
        """
        if getattr(self, "_faceDiffy", None) is None:
            self._faceDiffy = self.Pac.T * self.mesh.faceDivy * self.Pafy
        return self._faceDiffy

    @property
    def faceDiffz(self):
        """
        z-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the z-direction
        """
        if getattr(self, "_faceDiffz", None) is None:
            self._faceDiffz = self.Pac.T * self.mesh.faceDivz * self.Pafz
        return self._faceDiffz

    @property
    def cellDiffxStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the
        x-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, "_cellDiffxStencil", None) is None:

            self._cellDiffxStencil = (
                self.Pafx.T * self.mesh._cellGradxStencil * self.Pac
            )
        return self._cellDiffxStencil

    @property
    def cellDiffyStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the
        y-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if self.dim < 2:
            return None
        if getattr(self, "_cellDiffyStencil", None) is None:

            self._cellDiffyStencil = (
                self.Pafy.T * self.mesh._cellGradyStencil * self.Pac
            )
        return self._cellDiffyStencil

    @property
    def cellDiffzStencil(self):
        """
        cell centered difference stencil (no cell lengths include) in the
        y-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if self.dim < 3:
            return None
        if getattr(self, "_cellDiffzStencil", None) is None:

            self._cellDiffzStencil = (
                self.Pafz.T * self.mesh._cellGradzStencil * self.Pac
            )
        return self._cellDiffzStencil


# Make it look like it's in the regularization module
RegularizationMesh.__module__ = "SimPEG.regularization"

class LCRegularizationMesh(RegularizationMesh):
    """
    **LCRegularization Mesh**

    :param list mesh: lit including two discretize meshes
    :param numpy.ndarray indActive: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells
    :param numpy.ndarray indActiveEdges: bool array, size nE, that is True where we have active edges. Used to reduce the operators so we regularize only on active edges

    """

    def __init__(self, mesh, **kwargs):
        self.mesh_radial = mesh[0]
        self.mesh_vertical = mesh[1]
        utils.setKwargs(self, **kwargs)

    indActive = properties.Array("active indices in mesh", dtype=[bool, int])
    indActiveEdges = properties.Array(
        "indices of active edges in the mesh", dtype=(bool, int)
    )

    @properties.validator("indActive")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

    @properties.validator("indActiveEdges")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.nE, dtype=bool)
                value[tmp] = True
                change["value"] = value
    @property
    def vol(self):
        # Assume a unit area for the radial points)
        # We could use the average of cells to nodes
        self._vol = (np.ones(self.n_nodes, dtype=float)[:, None] * self.mesh_vertical.h[0]).flatten()
        return self._vol[self.indActive].flatten()

    @property
    def h_gridded_r(self):
        """
        Length of cells in the raidal direction

        """
        if getattr(self, "_h_gridded_r", None) is None:
            # assume a unit length scale in radial direction
            n = self.nz * self.n_nodes
            self._h_gridded_r = np.ones(n)
        return self._h_gridded_r

    @property
    def h_gridded_z(self):
        """
        Length of cells in the vertical direction

        """
        if getattr(self, "_h_gridded_z", None) is None:
            self._h_gridded_z = np.tile(
                self.mesh_vertical.h[0], self.n_nodes
            ).flatten()
        return self._h_gridded_z

    @property
    def nodal_gradient_stencil(self):
        ind_ptr = 2 * np.arange(self.mesh_radial.n_edges+1)
        col_inds = self.mesh_radial._edges.reshape(-1)
        Aijs = (np.ones(self.mesh_radial.n_edges, dtype=float)[:, None] * [-1, 1]).reshape(-1)

        return sp.csr_matrix((Aijs, col_inds, ind_ptr), shape=(self.mesh_radial.n_edges, self.n_nodes))

    @property
    def gradient_r(self):
        """
        Nodal gradient in radial direction

        """
        if getattr(self, "_gradient_r", None) is None:
            grad = self.nodal_gradient_stencil
            self._gradient_r = self.Paer.T * sp.kron(grad, utils.speye(self.nz)) * self.Pac
        return self._gradient_r

    @property
    def average_r(self):
        """
        Average of cells in the radial direction

        """
        if getattr(self, "_average_r", None) is None:
            ave = self.mesh_radial.average_node_to_edge
            self._average_r = self.Paer.T * sp.kron(ave, utils.speye(self.nz)) * self.Pac
        return self._average_r

    @property
    def gradient_z(self):
        """
        Cell gradeint in vertical direction

        """
        if getattr(self, "_gradient_z", None) is None:
            grad = self.mesh_vertical.stencil_cell_gradient
            self._gradient_z = self.Pafz.T * sp.kron(utils.speye(self.n_nodes), grad) * self.Pac
        return self._gradient_z

    @property
    def average_z(self):
        """
        Average of cells in the vertical direction

        """
        if getattr(self, "_average_z", None) is None:
            ave = self.mesh_vertical.average_cell_to_face
            self._average_z = self.Pafz.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pac
        return self._average_z

    @property
    def nz(self):
        """
        Number of cells of the 1D vertical mesh
        """
        if getattr(self, "_nz", None) is None:
            self._nz = self.mesh_vertical.n_cells
        return self._nz

    @property
    def nFz(self):
        """
        Number of faces in the vertical direction
        """
        if getattr(self, "_nFz", None) is None:
            self._nFz = self.mesh_vertical.n_faces * self.n_nodes
        return self._nFz

    @property
    def nE(self):
        """
        Number of edges in the radial direction
        """
        if getattr(self, "_nE", None) is None:
            self._nE = self.nz * self.n_edges
        return self._nE

    @property
    def nC(self):
        """
        reduced number of cells

        :rtype: int
        :return: number of cells being regularized
        """
        if self.indActive is not None:
            return int(self.indActive.sum())
        return self.nz * self.n_nodes

    @property
    def n_nodes(self):
        """
        Number of nodes of the 2D simplex mesh
        """
        if getattr(self, "_n_nodes", None) is None:
            self._n_nodes = self.mesh_radial.n_nodes
        return self._n_nodes

    @property
    def n_edges(self):
        """
        Number of edges of the 2D simplex mesh
        """
        if getattr(self, "_n_edges", None) is None:
            self._n_edges = self.mesh_radial.n_edges
        return self._n_edges

    @property
    def Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nindActive_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, "_Pafz", None) is None:
            if self.indActive is None:
                self._Pafz = utils.speye(self.nFz)
            else:
                ave = self.mesh_vertical.average_face_to_cell
                aveFz2CC = sp.kron(utils.speye(self.n_nodes), ave)
                indActive_Fz = aveFz2CC.T * self.indActive >= 1
                self._Pafz = utils.speye(self.nFz)[:, indActive_Fz]
        return self._Pafz

    @property
    def Pac(self):
        """
        projection matrix that takes from the reduced space of active cells to
        full modelling space (ie. nC x nindActive)

        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.indActive is None:
                self._Pac = utils.speye(self.nz*self.n_nodes)
            else:
                self._Pac = utils.speye(self.nz*self.n_nodes)[:, self.indActive]
        return self._Pac

    @property
    def Paer(self):
        """
        projection matrix that takes from the reduced space of active edges
        to full modelling space (ie. nE x nindActive_E )

        :rtype: scipy.sparse.csr_matrix
        :return: active edge projection matrix
        """
        if getattr(self, "_Paer", None) is None:
            if self.indActiveEdges is None:
                self._Paer = utils.speye(self.nE)
            else:
                ave = self.mesh_vertical.average_face_to_cell
                aveFz2CC = sp.kron(utils.speye(self.n_nodes), ave)
                self._Paer = utils.speye(self.nE)[:, self.indActiveEdges]
        return self._Paer

    @property
    def aveFz2CC(self):
        """
        averaging from active cell centers to active x-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active x-faces
        """
        if getattr(self, "_aveFz2CC", None) is None:
            ave = self.mesh_vertical.average_face_to_cell
            self._aveFz2CC = self.Pac.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pafz
        return self._aveFz2CC

    @property
    def aveE2N(self):
        """
        averaging from active nodes to active edges

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active edges
        """

        if getattr(self, "_aveE2N", None) is None:
            ave = self.mesh_radial.average_node_to_edge.T
            self._aveE2N = self.Pac.T * sp.kron(ave, utils.speye(self.nz)) * self.Paer
        return self._aveE2N
LCRegularizationMesh.__module__ = "SimPEG.regularization"