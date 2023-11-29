import numpy as np
import scipy.sparse as sp
from SimPEG.utils.code_utils import validate_active_indices

from .. import utils
from .regularization_mesh import RegularizationMesh


class LCRegularizationMesh(RegularizationMesh):
    """
    **LCRegularization Mesh**

    :param list mesh: lit including two discretize meshes
    :param numpy.ndarray active_cells: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells
    :param numpy.ndarray active_edges: bool array, size nE, that is True where we have active edges. Used to reduce the operators so we regularize only on active edges

    """

    _active_edges = None

    def __init__(self, mesh, active_cells=None, active_edges=None, **kwargs):
        self.mesh_radial = mesh[0]
        self.mesh_vertical = mesh[1]
        self.active_edges = active_edges
        utils.setKwargs(self, **kwargs)

    @property
    def active_cells(self) -> np.ndarray:
        """Active cells on the regularization mesh.

        A boolean array defining the cells in the regularization mesh that are active
        (i.e. updated) throughout the inversion. The values of inactive cells
        remain equal to their starting model values.

        Returns
        -------
        (n_cells, ) array of bool

        Notes
        -----
        If the property is set using a ``numpy.ndarray`` of ``int``, the setter interprets the
        array as representing the indices of the active cells. When called however, the quantity
        will have been internally converted to a boolean array.
        """
        return self._active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        if getattr(self, "_active_cells", None) is not None and not all(
            self._active_cells == values
        ):
            raise AttributeError(
                "The RegulatizationMesh already has an 'active_cells' property set."
            )
        if values is not None:
            values = validate_active_indices("values", values, self.nC)
            # Ensure any cached operators created when
            # active_cells was None are deleted
            self._vol = None
            self._Pac = None
            self._Paer = None
            self._Pafz = None
            self._h_gridded_r = None
            self._h_gridded_z = None
            self._cell_gradient_z = None
            self._aveCC2Fz = None
            self._aveFz2CC = None
        self._active_cells = values

    @property
    def active_edges(self) -> np.ndarray:
        return self._active_edges

    @active_edges.setter
    def active_edges(self, values: np.ndarray):
        if getattr(self, "_active_edges", None) is not None and not all(
            self._active_edges == values
        ):
            raise AttributeError(
                "The RegulatizationMesh already has an 'active_edges' property set."
            )
        if values is not None:
            self._aveCC2Fr = None
            self._cell_gradient_r = None
            self._aveFr2CC = None

        self._active_edges = values

    @property
    def vol(self) -> np.ndarray:
        # Assume a unit area for the radial points)
        # We could use the average of cells to nodes
        self._vol = (
            np.ones(self.n_nodes, dtype=float)[:, None] * self.mesh_vertical.h[0]
        ).flatten()
        return self._vol[self.active_cells].flatten()

    @property
    def h_gridded_r(self) -> np.ndarray:
        """
        Length of cells in the raidal direction

        """
        if getattr(self, "_h_gridded_r", None) is None:
            # assume a unit length scale in radial direction
            n = self.nz * self.n_nodes
            self._h_gridded_r = np.ones(n)
        return self._h_gridded_r

    @property
    def h_gridded_z(self) -> np.ndarray:
        """
        Length of cells in the vertical direction

        """
        if getattr(self, "_h_gridded_z", None) is None:
            self._h_gridded_z = np.tile(self.mesh_vertical.h[0], self.n_nodes).flatten()
        return self._h_gridded_z

    @property
    def base_length(self) -> float:
        """Smallest dimension (i.e. edge length) for smallest cell in the mesh.

        Returns
        -------
        float
            Smallest dimension (i.e. edge length) for smallest cell in the mesh.
        """
        if getattr(self, "_base_length", None) is None:
            self._base_length = self.mesh_vertical.h[0].min()
        return self._base_length

    @property
    def dim(self) -> int:
        """Dimension of regularization mesh.

        Returns
        -------
        {2}
            Dimension of the regularization mesh.
        """
        return 2

    @property
    def cell_gradient(self) -> sp.csr_matrix:
        """Cell gradient operator (cell centers to faces).

        Built from :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient`.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            Cell gradient operator (cell centers to faces).
        """
        return sp.vstack([self.cell_gradient_r, self.cell_gradient_z])

    @property
    def nodal_gradient_stencil(self) -> sp.csr_matrix:
        ind_ptr = 2 * np.arange(self.mesh_radial.n_edges + 1)
        col_inds = self.mesh_radial._edges.reshape(-1)
        Aijs = (
            np.ones(self.mesh_radial.n_edges, dtype=float)[:, None] * [-1, 1]
        ).reshape(-1)

        return sp.csr_matrix(
            (Aijs, col_inds, ind_ptr), shape=(self.mesh_radial.n_edges, self.n_nodes)
        )

    @property
    def cell_gradient_r(self) -> sp.csr_matrix:
        """
        Nodal gradient in radial direction

        """
        if getattr(self, "_cell_gradient_r", None) is None:
            grad = self.nodal_gradient_stencil
            self._cell_gradient_r = (
                self.Paer.T * sp.kron(grad, utils.speye(self.nz)) * self.Pac
            )
        return self._cell_gradient_r

    @property
    def aveCC2Fr(self) -> sp.csr_matrix:
        """
        Average of cells in the radial direction

        """
        if getattr(self, "_aveCC2Fr", None) is None:
            ave = self.mesh_radial.average_node_to_edge
            self._aveCC2Fr = self.Paer.T * sp.kron(ave, utils.speye(self.nz)) * self.Pac
        return self._aveCC2Fr

    @property
    def cell_distances_r(self) -> np.ndarray:
        """Cell center distance array along the r-direction.

        Returns
        -------
        (n_active_faces_r, ) numpy.ndarray
            Cell center distance array along the r-direction.
        """
        if getattr(self, "_cell_distances_r", None) is None:
            Ave = self.aveCC2Fr
            self._cell_distances_r = Ave * (self.Pac.T * self.h_gridded_r)
        return self._cell_distances_r

    @property
    def cell_gradient_z(self) -> sp.csr_matrix:
        """
        Cell gradeint in vertical direction

        """
        if getattr(self, "_cell_gradient_z", None) is None:
            grad = self.mesh_vertical.stencil_cell_gradient
            self._cell_gradient_z = (
                self.Pafz.T * sp.kron(utils.speye(self.n_nodes), grad) * self.Pac
            )
        return self._cell_gradient_z

    @property
    def aveCC2Fz(self) -> sp.csr_matrix:
        """
        Average of cells in the vertical direction

        """
        if getattr(self, "_aveCC2Fz", None) is None:
            ave = self.mesh_vertical.average_cell_to_face
            self._aveCC2Fz = (
                self.Pafz.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pac
            )
        return self._aveCC2Fz

    @property
    def cell_distances_z(self) -> np.ndarray:
        """Cell center distance array along the r-direction.

        Returns
        -------
        (n_active_faces_z, ) numpy.ndarray
            Cell center distance array along the r-direction.
        """
        if getattr(self, "_cell_distances_z", None) is None:
            Ave = self.aveCC2Fr
            self._cell_distances_z = Ave * (self.Pac.T * self.h_gridded_z)
        return self._cell_distances_z

    @property
    def nz(self) -> int:
        """
        Number of cells of the 1D vertical mesh
        """
        if getattr(self, "_nz", None) is None:
            self._nz = self.mesh_vertical.n_cells
        return self._nz

    @property
    def nFz(self) -> int:
        """
        Number of faces in the vertical direction
        """
        if getattr(self, "_nFz", None) is None:
            self._nFz = self.mesh_vertical.n_faces * self.n_nodes
        return self._nFz

    @property
    def nE(self) -> int:
        """
        Number of edges in the radial direction
        """
        if getattr(self, "_nE", None) is None:
            self._nE = self.nz * self.n_edges
        return self._nE

    @property
    def nC(self) -> int:
        """
        reduced number of cells

        :rtype: int
        :return: number of cells being regularized
        """
        if self.active_cells is not None:
            return int(self.active_cells.sum())
        return self.nz * self.n_nodes

    @property
    def n_nodes(self) -> int:
        """
        Number of nodes of the 2D simplex mesh
        """
        if getattr(self, "_n_nodes", None) is None:
            self._n_nodes = self.mesh_radial.n_nodes
        return self._n_nodes

    @property
    def n_edges(self) -> int:
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
        to full modelling space (ie. nFz x nactive_cells_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, "_Pafz", None) is None:
            if self.active_cells is None:
                self._Pafz = utils.speye(self.nFz)
            else:
                ave = self.mesh_vertical.average_face_to_cell
                aveFz2CC = sp.kron(utils.speye(self.n_nodes), ave)
                active_cells_Fz = aveFz2CC.T * self.active_cells >= 1
                self._Pafz = utils.speye(self.nFz)[:, active_cells_Fz]
        return self._Pafz

    @property
    def Pac(self):
        """
        projection matrix that takes from the reduced space of active cells to
        full modelling space (ie. nC x nactive_cells)

        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.active_cells is None:
                self._Pac = utils.speye(self.nz * self.n_nodes)
            else:
                self._Pac = utils.speye(self.nz * self.n_nodes)[:, self.active_cells]
        return self._Pac

    @property
    def Paer(self):
        """
        projection matrix that takes from the reduced space of active edges
        to full modelling space (ie. nE x nactive_cells_E )

        :rtype: scipy.sparse.csr_matrix
        :return: active edge projection matrix
        """
        if getattr(self, "_Paer", None) is None:
            if self.active_edges is None:
                self._Paer = utils.speye(self.nE)
            else:
                ave = self.mesh_vertical.average_face_to_cell
                self._Paer = utils.speye(self.nE)[:, self.active_edges]
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
            self._aveFz2CC = (
                self.Pac.T * sp.kron(utils.speye(self.n_nodes), ave) * self.Pafz
            )
        return self._aveFz2CC

    @property
    def aveFr2CC(self):
        """
        averaging from active nodes to active edges

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active edges
        """

        if getattr(self, "_aveFr2CC", None) is None:
            ave = self.mesh_radial.average_node_to_edge.T
            self._aveFr2CC = self.Pac.T * sp.kron(ave, utils.speye(self.nz)) * self.Paer
        return self._aveFr2CC


LCRegularizationMesh.__module__ = "SimPEG.regularization"
