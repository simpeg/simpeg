import numpy as np
import scipy.sparse as sp

from simpeg.utils.code_utils import deprecate_property, validate_active_indices

from .. import props, utils

###############################################################################
#                                                                             #
#                             Regularization Mesh                             #
#                                                                             #
###############################################################################


class RegularizationMesh(props.BaseSimPEG):
    """Regularization Mesh

    The ``RegularizationMesh`` class is used to construct differencing and averaging operators
    for the objective function(s) defining the regularization. In practice, these operators are
    not constructed by creating instances of ``RegularizationMesh``. The operators are instead
    constructed (and sometimes stored) when called as a property of the mesh.
    The ``RegularizationMesh`` class is built using much of the functionality from the
    :py:class:`discretize.operators.differential_operators.DiffOperators` class.
    However, operators constructed using the ``RegularizationMesh`` class have been modified to
    act only on interior faces and active cells in the inversion, thus reducing computational cost.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        Mesh on which the discrete set of model parameters are defined.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of mesh cells that are active in the inversion.
        If ``None``, all cells are active.
    """

    regularization_type = None  # or 'Base'
    _active_cells = None

    def __init__(self, mesh, active_cells=None, **kwargs):
        self.mesh = mesh
        self.active_cells = active_cells
        utils.set_kwargs(self, **kwargs)

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
            values = validate_active_indices("values", values, self.mesh.nC)
            # Ensure any cached operators created when
            # active_cells was None are deleted
            self._vol = None
            self._Pac = None
            self._Pafx = None
            self._Pafy = None
            self._Pafz = None
            self._aveFx2CC = None
            self._aveFy2CC = None
            self._aveFz2CC = None
            self._aveCC2Fx = None
            self._aveCC2Fy = None
            self._aveCC2Fz = None
            self._cell_gradient_x = None
            self._cell_gradient_y = None
            self._cell_gradient_z = None
            self._faceDiffx = None
            self._faceDiffy = None
            self._faceDiffz = None
            self._cell_distances_x = None
            self._cell_distances_y = None
            self._cell_distances_z = None
        self._active_cells = values

    @property
    def vol(self) -> np.ndarray:
        """Volumes of active mesh cells.

        Returns
        -------
        (n_active, ) numpy.ndarray of float
            Volumes of active mesh cells.
        """
        if self.active_cells is None:
            return self.mesh.cell_volumes
        if getattr(self, "_vol", None) is None:
            self._vol = self.mesh.cell_volumes[self.active_cells]
        return self._vol

    @property
    def n_cells(self) -> int:
        """Number of active cells.

        Returns
        -------
        int
            Number of active cells.
        """
        if self.active_cells is not None:
            return int(self.active_cells.sum())
        return self.mesh.n_cells

    nC = n_cells

    @property
    def dim(self) -> int:
        """Dimension of regularization mesh.

        Returns
        -------
        {1, 2, 3}
            Dimension of the regularization mesh.
        """
        return self.mesh.dim

    @property
    def Pac(self) -> sp.csr_matrix:
        """Projection matrix from active cells to all mesh cells.

        Returns
        -------
        (n_cells, n_active) scipy.sparse.csr_matrix
            Projection matrix from active cells to all mesh cells.
        """
        if getattr(self, "_Pac", None) is None:
            if self.active_cells is None:
                self._Pac = utils.speye(self.mesh.nC)
            else:
                self._Pac = utils.speye(self.mesh.nC)[:, self.active_cells]
        return self._Pac

    @property
    def Pafx(self) -> sp.csr_matrix:
        """Projection matrix from active x-faces to all x-faces in the mesh.

        Returns
        -------
        (n_faces_x, n_active_faces_x) scipy.sparse.csr_matrix
            Projection matrix from active x-faces to all x-faces in the mesh.
        """
        if getattr(self, "_Pafx", None) is None:
            if self.mesh._meshType == "TREE":
                ind_active = self.active_cells
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                active_cells_Fx = (
                    self.mesh.average_cell_to_total_face_x() * ind_active
                ) >= 1
                self._Pafx = utils.speye(self.mesh.ntFx)[:, active_cells_Fx]
            else:
                if self.active_cells is None:
                    self._Pafx = utils.speye(self.mesh.nFx)
                else:
                    active_cells_Fx = self.mesh.aveFx2CC.T * self.active_cells >= 1
                    self._Pafx = utils.speye(self.mesh.nFx)[:, active_cells_Fx]
        return self._Pafx

    @property
    def Pafy(self) -> sp.csr_matrix:
        """Projection matrix from active y-faces to all y-faces in the mesh.

        Returns
        -------
        (n_faces_y, n_active_faces_y) scipy.sparse.csr_matrix
            Projection matrix from active y-faces to all y-faces in the mesh.
        """
        if getattr(self, "_Pafy", None) is None:
            if self.mesh._meshType == "TREE":
                ind_active = self.active_cells
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                active_cells_Fy = (
                    self.mesh.average_cell_to_total_face_y() * ind_active
                ) >= 1
                self._Pafy = utils.speye(self.mesh.ntFy)[:, active_cells_Fy]
            elif self.mesh._meshType == "CYL" and self.mesh.is_symmetric:
                return sp.csr_matrix((0, 0))
            else:
                if self.active_cells is None:
                    self._Pafy = utils.speye(self.mesh.nFy)
                else:
                    active_cells_Fy = (self.mesh.aveFy2CC.T * self.active_cells) >= 1
                    self._Pafy = utils.speye(self.mesh.nFy)[:, active_cells_Fy]
        return self._Pafy

    @property
    def Pafz(self) -> sp.csr_matrix:
        """Projection matrix from active z-faces to all z-faces in the mesh.

        Returns
        -------
        (n_faces_z, n_active_faces_z) scipy.sparse.csr_matrix
            Projection matrix from active z-faces to all z-faces in the mesh.
        """
        if getattr(self, "_Pafz", None) is None:
            if self.mesh._meshType == "TREE":
                ind_active = self.active_cells
                if ind_active is None:
                    ind_active = np.ones(self.mesh.nC, dtype="bool")
                active_cells_Fz = (
                    self.mesh.average_cell_to_total_face_z() * ind_active
                ) >= 1
                self._Pafz = utils.speye(self.mesh.ntFz)[:, active_cells_Fz]
            else:
                if self.active_cells is None:
                    self._Pafz = utils.speye(self.mesh.nFz)
                else:
                    active_cells_Fz = (self.mesh.aveFz2CC.T * self.active_cells) >= 1
                    self._Pafz = utils.speye(self.mesh.nFz)[:, active_cells_Fz]
        return self._Pafz

    @property
    def average_face_to_cell(self) -> sp.csr_matrix:
        """Averaging operator from faces to cell centers.

        Built from :py:property:`~discretize.operators.differential_operators.DiffOperators.average_face_to_cell`.

        Returns
        -------
        (n_cells, n_faces) scipy.sparse.csr_matrix
            Averaging operator from faces to cell centers.
        """
        if self.dim == 1:
            return self.aveFx2CC
        elif self.dim == 2:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC])
        else:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC, self.aveFz2CC])

    @property
    def aveFx2CC(self) -> sp.csr_matrix:
        """Averaging operator from active cell centers to active x-faces.

        Modified from the transpose of
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveFx2CC`;
        an operator that projects from all x-faces to all cell centers.

        Returns
        -------
        (n_active_cells, n_active_faces_x) scipy.sparse.csr_matrix
            Averaging operator from active cell centers to active x-faces.
        """
        if getattr(self, "_aveFx2CC", None) is None:
            if self.mesh._meshType == "TREE":
                nCinRow = utils.mkvc((self.aveCC2Fx.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFx2CC = utils.sdiag(nCinRow) * self.aveCC2Fx.T
            else:
                self._aveFx2CC = self.Pac.T * self.mesh.aveFx2CC * self.Pafx

        return self._aveFx2CC

    @property
    def aveCC2Fx(self) -> sp.csr_matrix:
        """Averaging operator from active x-faces to active cell centers.

        Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveCC2Fx`;
        an operator that projects from all x-faces to all cell centers.

        Returns
        -------
        (n_active_faces_x, n_active_cells) scipy.sparse.csr_matrix
            Averaging operator from active x-faces to active cell centers.
        """
        if getattr(self, "_aveCC2Fx", None) is None:
            if self.mesh._meshType == "TREE":
                self._aveCC2Fx = (
                    self.Pafx.T * self.mesh.average_cell_to_total_face_x() * self.Pac
                )
            else:
                self._aveCC2Fx = (
                    utils.sdiag(1.0 / (self.aveFx2CC.T).sum(1)) * self.aveFx2CC.T
                )
        return self._aveCC2Fx

    @property
    def aveFy2CC(self) -> sp.csr_matrix:
        """Averaging operator from active cell centers to active y-faces.

        Modified from the transpose of
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveFy2CC`;
        an operator that projects from y-faces to cell centers.

        Returns
        -------
        (n_active_cells, n_active_faces_y) scipy.sparse.csr_matrix
            Averaging operator from active cell centers to active y-faces.
        """
        if getattr(self, "_aveFy2CC", None) is None:
            if self.mesh._meshType == "TREE":
                nCinRow = utils.mkvc((self.aveCC2Fy.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFy2CC = utils.sdiag(nCinRow) * self.aveCC2Fy.T
            elif self.mesh._meshType == "CYL" and self.mesh.is_symmetric:
                return sp.csr_matrix((self.nC, 0))
            else:
                self._aveFy2CC = self.Pac.T * self.mesh.aveFy2CC * self.Pafy

        return self._aveFy2CC

    @property
    def aveCC2Fy(self) -> sp.csr_matrix:
        """Averaging operator from active y-faces to active cell centers.

        Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveCC2Fy`;
        an operator that projects from all y-faces to all cell centers.

        Returns
        -------
        (n_active_faces_y, n_active_cells) scipy.sparse.csr_matrix
            Averaging operator from active y-faces to active cell centers.
        """
        if getattr(self, "_aveCC2Fy", None) is None:
            if self.mesh._meshType == "TREE":
                self._aveCC2Fy = (
                    self.Pafy.T * self.mesh.average_cell_to_total_face_y() * self.Pac
                )
            elif self.mesh._meshType == "CYL" and self.mesh.is_symmetric:
                return sp.csr_matrix((0, self.nC))
            else:
                self._aveCC2Fy = (
                    utils.sdiag(1.0 / (self.aveFy2CC.T).sum(1)) * self.aveFy2CC.T
                )
        return self._aveCC2Fy

    @property
    def aveFz2CC(self) -> sp.csr_matrix:
        """Averaging operator from active cell centers to active z-faces.

        Modified from the transpose of
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveFz2CC`;
        an operator that projects from z-faces to cell centers.

        Returns
        -------
        (n_active_cells, n_active_faces_z) scipy.sparse.csr_matrix
            Averaging operator from active cell centers to active z-faces.
        """
        if getattr(self, "_aveFz2CC", None) is None:
            if self.mesh._meshType == "TREE":
                nCinRow = utils.mkvc((self.aveCC2Fz.T).sum(1))
                nCinRow[nCinRow > 0] = 1.0 / nCinRow[nCinRow > 0]
                self._aveFz2CC = utils.sdiag(nCinRow) * self.aveCC2Fz.T
            else:
                self._aveFz2CC = self.Pac.T * self.mesh.aveFz2CC * self.Pafz

        return self._aveFz2CC

    @property
    def aveCC2Fz(self) -> sp.csr_matrix:
        """Averaging operator from active z-faces to active cell centers.

        Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.aveCC2Fz`;
        an operator that projects from all z-faces to all cell centers.

        Returns
        -------
        (n_active_faces_z, n_active_cells) scipy.sparse.csr_matrix
            Averaging operator from active z-faces to active cell centers.
        """
        if getattr(self, "_aveCC2Fz", None) is None:
            if self.mesh._meshType == "TREE":
                self._aveCC2Fz = (
                    self.Pafz.T * self.mesh.average_cell_to_total_face_z() * self.Pac
                )
            else:
                self._aveCC2Fz = (
                    utils.sdiag(1.0 / (self.aveFz2CC.T).sum(1)) * self.aveFz2CC.T
                )
        return self._aveCC2Fz

    @property
    def base_length(self) -> float:
        """Smallest dimension (i.e. edge length) for smallest cell in the mesh.

        Returns
        -------
        float
            Smallest dimension (i.e. edge length) for smallest cell in the mesh.
        """
        if getattr(self, "_base_length", None) is None:
            self._base_length = self.mesh.edge_lengths.min()
        return self._base_length

    @property
    def cell_gradient(self) -> sp.csr_matrix:
        """Cell gradient operator (cell centers to faces).

        Built from :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient`.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            Cell gradient operator (cell centers to faces).
        """
        if self.dim == 1:
            return self.cell_gradient_x
        elif self.dim == 2:
            return sp.vstack([self.cell_gradient_x, self.cell_gradient_y])
        else:
            return sp.vstack(
                [self.cell_gradient_x, self.cell_gradient_y, self.cell_gradient_z]
            )

    @property
    def cell_gradient_x(self) -> sp.csr_matrix:
        """Cell-centered x-derivative operator on active cells.

        Cell centered x-derivative operator that maps from active cells
        to active x-faces. Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient_x`.

        Returns
        -------
        (n_active_faces_x, n_active_cells) scipy.sparse.csr_matrix
            Cell-centered x-derivative operator on active cells.
        """
        if getattr(self, "_cell_gradient_x", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_x = (
                    self.Pafx.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_x()
                        * (self.mesh.h_gridded[:, 0] ** -1)
                    )
                    * self.mesh.stencil_cell_gradient_x
                    * self.Pac
                )
            else:
                self._cell_gradient_x = (
                    self.Pafx.T * self.mesh.cell_gradient_x * self.Pac
                )
        return self._cell_gradient_x

    @property
    def cell_gradient_y(self) -> sp.csr_matrix:
        """Cell-centered y-derivative operator on active cells.

        Cell centered y-derivative operator that maps from active cells
        to active y-faces. Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient_y`.

        Returns
        -------
        (n_active_faces_y, n_active_cells) scipy.sparse.csr_matrix
            Cell-centered y-derivative operator on active cells.
        """
        if getattr(self, "_cell_gradient_y", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_y = (
                    self.Pafy.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_y()
                        * (self.mesh.h_gridded[:, 1] ** -1)
                    )
                    * self.mesh.stencil_cell_gradient_y
                    * self.Pac
                )
            else:
                self._cell_gradient_y = (
                    self.Pafy.T * self.mesh.cell_gradient_y * self.Pac
                )
        return self._cell_gradient_y

    @property
    def cell_gradient_z(self) -> sp.csr_matrix:
        """Cell-centered z-derivative operator on active cells.

        Cell centered z-derivative operator that maps from active cells
        to active z-faces. Modified from
        :py:property:`~discretize.operators.differential_operators.DiffOperators.cell_gradient_z`.

        Returns
        -------
        (n_active_faces_z, n_active_cells) scipy.sparse.csr_matrix
            Cell-centered z-derivative operator on active cells.
        """
        if getattr(self, "_cell_gradient_z", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_z = (
                    self.Pafz.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_z()
                        * (self.mesh.h_gridded[:, 2] ** -1)
                    )
                    * self.mesh.stencil_cell_gradient_z
                    * self.Pac
                )
            else:
                self._cell_gradient_z = (
                    self.Pafz.T * self.mesh.cell_gradient_z * self.Pac
                )
        return self._cell_gradient_z

    cellDiffx = deprecate_property(
        cell_gradient_x,
        "cellDiffx",
        "cell_gradient_x",
        "0.19.0",
        error=True,
    )
    cellDiffy = deprecate_property(
        cell_gradient_y,
        "cellDiffy",
        "cell_gradient_y",
        "0.19.0",
        error=True,
    )
    cellDiffz = deprecate_property(
        cell_gradient_z,
        "cellDiffz",
        "cell_gradient_z",
        "0.19.0",
        error=True,
    )

    @property
    def cell_distances_x(self) -> np.ndarray:
        """Cell center distance array along the x-direction.

        Returns
        -------
        (n_active_faces_x, ) numpy.ndarray
            Cell center distance array along the x-direction.
        """
        if getattr(self, "_cell_distances_x", None) is None:
            self._cell_distances_x = self.cell_gradient_x.max(
                axis=1
            ).toarray().ravel() ** (-1.0)

        return self._cell_distances_x

    @property
    def cell_distances_y(self) -> np.ndarray:
        """Cell center distance array along the y-direction.

        Returns
        -------
        (n_active_faces_y, ) numpy.ndarray
            Cell center distance array along the y-direction.
        """
        if getattr(self, "_cell_distances_y", None) is None:
            self._cell_distances_y = self.cell_gradient_y.max(
                axis=1
            ).toarray().ravel() ** (-1.0)

        return self._cell_distances_y

    @property
    def cell_distances_z(self) -> np.ndarray:
        """Cell center distance array along the z-direction.

        Returns
        -------
        (n_active_faces_z, ) numpy.ndarray
            Cell center distance array along the z-direction.
        """
        if getattr(self, "_cell_distances_z", None) is None:
            self._cell_distances_z = self.cell_gradient_z.max(
                axis=1
            ).toarray().ravel() ** (-1.0)

        return self._cell_distances_z


# Make it look like it's in the regularization module
RegularizationMesh.__module__ = "simpeg.regularization"
