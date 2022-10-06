import numpy as np
import scipy.sparse as sp
from SimPEG.utils.code_utils import deprecate_property

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
    :param numpy.ndarray active_cells: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells

    """

    regularization_type = None  # or 'Base'
    _active_cells = None

    def __init__(self, mesh, active_cells=None, **kwargs):
        self.mesh = mesh
        self.active_cells = active_cells
        utils.setKwargs(self, **kwargs)

    @property
    def active_cells(self) -> np.ndarray:
        """A boolean array indicating whether a cell is active

        Notes
        -----
        If this is set with an array of integers, it interprets it as an array
        listing the active cell indices.
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
            try:
                values = np.asarray(values)
            except:
                raise ValueError("Input 'active_cells' must be array_like.")

            if values.dtype != bool:
                try:
                    tmp = np.zeros(self.mesh.nC, dtype=bool)
                    tmp[values] = True
                except:
                    raise ValueError(
                        "Values must be an array of integers or an array of bools "
                        "indicating the active cells"
                    )
                if np.sum(tmp) != len(values):
                    # This line should cause an error to be thrown if someone
                    # accidentally passes a list of 0 & 1 integers instead of passing
                    # it a list of booleans.
                    raise ValueError(
                        "Array was interpretted as a list of active indices and you "
                        "attempted to set the same cell as active multiple times."
                    )
                values = tmp

            if values.shape != (self.mesh.nC,):
                raise ValueError(
                    f"Input 'active_cells' must have shape {(self.mesh.nC,)}"
                )
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
        """
        Reduced volume vector.
        """
        if self.active_cells is None:
            return self.mesh.cell_volumes
        if getattr(self, "_vol", None) is None:
            self._vol = self.mesh.cell_volumes[self.active_cells]
        return self._vol

    @property
    def nC(self) -> int:
        """
        Number of cells being regularized.
        """
        if self.active_cells is not None:
            return int(self.active_cells.sum())
        return self.mesh.nC

    @property
    def dim(self) -> int:
        """
        Dimension of regularization mesh (1D, 2D, 3D)
        """
        return self.mesh.dim

    @property
    def Pac(self) -> sp.csr_matrix:
        """
        Projection matrix that takes from the reduced space of active cells to
        full modelling space (ie. nC x nactive_cells).
        """
        if getattr(self, "_Pac", None) is None:
            if self.active_cells is None:
                self._Pac = utils.speye(self.mesh.nC)
            else:
                self._Pac = utils.speye(self.mesh.nC)[:, self.active_cells]
        return self._Pac

    @property
    def Pafx(self) -> sp.csr_matrix:
        """
        Projection matrix that takes from the reduced space of active x-faces
        to full modelling space (ie. nFx x nactive_cells_Fx )
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
        """
        Projection matrix that takes from the reduced space of active y-faces
        to full modelling space (ie. nFy x nactive_cells_Fy ).
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
        """
        Projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nactive_cells_Fz ).
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
        """
        Vertically stacked matrix of cell averaging operators from active
        cell centers to active faces along each dimension of the mesh.
        """
        if self.dim == 1:
            return self.aveFx2CC
        elif self.dim == 2:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC])
        else:
            return sp.hstack([self.aveFx2CC, self.aveFy2CC, self.aveFz2CC])

    @property
    def aveFx2CC(self) -> sp.csr_matrix:
        """
        Averaging from active cell centers to active x-faces.
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
        """
        Averaging from active x-faces to active cell centers.
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
        """
        Averaging from active cell centers to active y-faces.
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
        """
        Averaging matrix from active y-faces to active cell centers.
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
        """
        Averaging from active cell centers to active z-faces.
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
        """
        Averaging matrix from active z-faces to active cell centers.
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
        """The smallest core cell size."""
        if getattr(self, "_base_length", None) is None:
            self._base_length = self.mesh.edge_lengths.min()
        return self._base_length

    @property
    def cell_gradient(self) -> sp.csr_matrix:
        """
        Vertically stacked matrix of cell gradients along each dimension of
        the mesh.
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
        """
        Cell centered gradient matrix for active cells in the x-direction.
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
        """
        Cell centered gradient matrix for active cells in the y-direction.
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
        """
        Cell centered gradient matrix for active cells in the z-direction.
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
        cell_gradient_x, "cellDiffx", "0.x.0", error=False, future_warn=False
    )
    cellDiffy = deprecate_property(
        cell_gradient_y, "cellDiffy", "0.x.0", error=False, future_warn=False
    )
    cellDiffz = deprecate_property(
        cell_gradient_z, "cellDiffz", "0.x.0", error=False, future_warn=False
    )

    @property
    def cell_distances_x(self) -> np.ndarray:
        """
        Cell center distance array along the x-direction.
        """
        if getattr(self, "_cell_distances_x", None) is None:
            Ave = self.aveCC2Fx
            self._cell_distances_x = Ave * (self.Pac.T * self.mesh.h_gridded[:, 0])
        return self._cell_distances_x

    @property
    def cell_distances_y(self) -> np.ndarray:
        """
        Cell center distance array along the y-direction.
        """
        if getattr(self, "_cell_distances_y", None) is None:
            Ave = self.aveCC2Fy
            self._cell_distances_y = Ave * (self.Pac.T * self.mesh.h_gridded[:, 1])
        return self._cell_distances_y

    @property
    def cell_distances_z(self) -> np.ndarray:
        """
        Cell center distance array along the z-direction.
        """
        if getattr(self, "_cell_distances_z", None) is None:
            Ave = self.aveCC2Fz
            self._cell_distances_z = Ave * (self.Pac.T * self.mesh.h_gridded[:, 2])
        return self._cell_distances_z


# Make it look like it's in the regularization module
RegularizationMesh.__module__ = "SimPEG.regularization"
