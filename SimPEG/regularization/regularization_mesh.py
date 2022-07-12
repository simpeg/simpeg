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

    def __init__(self, mesh, active_cells=None, **kwargs):
        self.mesh = mesh
        self.active_cells = active_cells
        utils.setKwargs(self, **kwargs)

    # active_cells = properties.Array("active indices in mesh", dtype=[bool, int])

    @property
    def active_cells(self):
        """A boolean array indicating whether a cell is active

        Returns
        -------
        (n_cells,) numpy.ndarray of bool

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
    def vol(self):
        """
        reduced volume vector

        :rtype: numpy.ndarray
        :return: reduced cell volume
        """
        if self.active_cells is None:
            return self.mesh.cell_volumes
        if getattr(self, "_vol", None) is None:
            self._vol = self.mesh.cell_volumes[self.active_cells]
        return self._vol

    @property
    def nC(self):
        """
        reduced number of cells

        :rtype: int
        :return: number of cells being regularized
        """
        if self.active_cells is not None:
            return int(self.active_cells.sum())
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
        full modelling space (ie. nC x nactive_cells)

        :rtype: scipy.sparse.csr_matrix
        :return: active cell projection matrix
        """
        if getattr(self, "_Pac", None) is None:
            if self.active_cells is None:
                self._Pac = utils.speye(self.mesh.nC)
            else:
                self._Pac = utils.speye(self.mesh.nC)[:, self.active_cells]
        return self._Pac

    @property
    def Pafx(self):
        """
        projection matrix that takes from the reduced space of active x-faces
        to full modelling space (ie. nFx x nactive_cells_Fx )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
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
    def Pafy(self):
        """
        projection matrix that takes from the reduced space of active y-faces
        to full modelling space (ie. nFy x nactive_cells_Fy )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-y projection matrix
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
            elif self.mesh_meshType == "CYL" and self.mesh.is_symmetric:
                return None
            else:
                if self.active_cells is None:
                    self._Pafy = utils.speye(self.mesh.nFy)
                else:
                    active_cells_Fy = (self.mesh.aveFy2CC.T * self.active_cells) >= 1
                    self._Pafy = utils.speye(self.mesh.nFy)[:, active_cells_Fy]
        return self._Pafy

    @property
    def Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nactive_cells_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-z projection matrix
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
            if self.mesh._meshType == "TREE":
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
    def aveFy2CC(self):
        """
        averaging from active cell centers to active y-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active y-faces
        """
        if getattr(self, "_aveFy2CC", None) is None:
            if self.mesh._meshType == "TREE":
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
            if self.mesh._meshType == "TREE":
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
            if self.mesh._meshType == "TREE":
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
    def base_length(self):
        """The smallest core cell size"""
        if getattr(self, "_base_length", None) is None:
            self._base_length = self.mesh.edge_lengths.min()
        return self._base_length

    @property
    def cell_gradient(self):
        if self.dim == 1:
            return self.cell_gradient_x
        elif self.dim == 2:
            return sp.vstack([self.cell_gradient_x, self.cell_gradient_y])
        else:
            return sp.vstack(
                [self.cell_gradient_x, self.cell_gradient_y, self.cell_gradient_z]
            )

    @property
    def cell_gradient_x(self):
        """
        cell centered gradient in the x-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, "_cell_gradient_x", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_x = (
                    self.Pafx.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_x()
                        * (self.mesh.h_gridded[:, 0] ** -1)
                    )
                    * self.mesh._cellGradxStencil
                    * self.Pac
                )
            else:
                self._cell_gradient_x = (
                    self.Pafx.T * self.mesh.cell_gradient_x * self.Pac
                )
        return self._cell_gradient_x

    @property
    def cell_gradient_y(self):
        """
        cell centered gradient in the y-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if getattr(self, "_cell_gradient_y", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_y = (
                    self.Pafy.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_y()
                        * (self.mesh.h_gridded[:, 1] ** -1)
                    )
                    * self.mesh._cellGradyStencil
                    * self.Pac
                )
            else:
                self._cell_gradient_y = (
                    self.Pafy.T * self.mesh.cell_gradient_y * self.Pac
                )
        return self._cell_gradient_y

    @property
    def cell_gradient_z(self):
        """
        cell centered difference in the z-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the z-direction
        """
        if getattr(self, "_cell_gradient_z", None) is None:
            if self.mesh._meshType == "TREE":
                self._cell_gradient_z = (
                    self.Pafz.T
                    * utils.sdiag(
                        self.mesh.average_cell_to_total_face_z()
                        * (self.mesh.h_gridded[:, 2] ** -1)
                    )
                    * self.mesh._cellGradzStencil
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
    def cell_distances_x(self):
        if getattr(self, "_cell_distances_x", None) is None:
            Ave = self.aveCC2Fx
            self._cell_distances_x = Ave * (self.Pac.T * self.mesh.h_gridded[:, 0])
        return self._cell_distances_x

    @property
    def cell_distances_y(self):
        if getattr(self, "_cell_distances_y", None) is None:
            Ave = self.aveCC2Fy
            self._cell_distances_y = Ave * (self.Pac.T * self.mesh.h_gridded[:, 1])
        return self._cell_distances_x

    @property
    def cell_distances_z(self):
        if getattr(self, "_cell_distances_z", None) is None:
            Ave = self.aveCC2Fz
            self._cell_distances_z = Ave * (self.Pac.T * self.mesh.h_gridded[:, 2])
        return self._cell_distances_x

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


# Make it look like it's in the regularization module
RegularizationMesh.__module__ = "SimPEG.regularization"
