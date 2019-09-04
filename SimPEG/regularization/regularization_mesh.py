import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .. import Props
from .. import Utils

###############################################################################
#                                                                             #
#                             Regularization Mesh                             #
#                                                                             #
###############################################################################

class RegularizationMesh(Props.BaseSimPEG):
    """
    **Regularization Mesh**

    This contains the operators used in the regularization. Note that these
    are not necessarily true differential operators, but are constructed from
    a SimPEG Mesh.

    :param discretize.base.BaseMesh mesh: problem mesh
    :param numpy.ndarray indActive: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells

    """

    regularization_type = None  # or 'Simple', 'Sparse' or 'Tikhonov'

    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        Utils.setKwargs(self, **kwargs)

    indActive = properties.Array("active indices in mesh", dtype=[bool, int])

    @properties.validator('indActive')
    def _cast_to_bool(self, change):
        value = change['value']
        if value is not None:
            if value.dtype != 'bool':  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
                value[tmp] = True
                change['value'] = value

    @property
    def vol(self):
        """
        reduced volume vector

        :rtype: numpy.ndarray
        :return: reduced cell volume
        """
        if getattr(self, '_vol', None) is None:
            self._vol = self.Pac.T * self.mesh.vol
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
        if getattr(self, '_dim', None) is None:
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
        if getattr(self, '_Pac', None) is None:
            if self.indActive is None:
                self._Pac = Utils.speye(self.mesh.nC)
            else:
                self._Pac = Utils.speye(self.mesh.nC)[:, self.indActive]
        return self._Pac

    @property
    def Pafx(self):
        """
        projection matrix that takes from the reduced space of active x-faces
        to full modelling space (ie. nFx x nindActive_Fx )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-x projection matrix
        """
        if getattr(self, '_Pafx', None) is None:
            if self.indActive is None:
                self._Pafx = Utils.speye(self.mesh.nFx)
            else:
                # if getattr(self.mesh, 'aveCC2Fx', None) is not None:
                if self.mesh._meshType == "TREE":
                    if self.regularization_type == "Tikhonov":
                        indActive_Fx = (
                            (self.mesh.aveFx2CC.T * self.indActive) >= 1
                        )
                        self._Pafx = (
                            Utils.speye(self.mesh.nFx)[:, indActive_Fx]
                        )
                    else:
                        indActive_Fx = (
                            (
                                self.mesh._aveCC2FxStencil() *
                                self.indActive
                            ) >= 1
                        )
                        self._Pafx = (
                            Utils.speye(self.mesh.ntFx)[:, indActive_Fx]
                        )
                else:
                    indActive_Fx = self.mesh.aveFx2CC.T * self.indActive >= 1

                    self._Pafx = Utils.speye(self.mesh.nFx)[:, indActive_Fx]
        return self._Pafx

    @property
    def Pafy(self):
        """
        projection matrix that takes from the reduced space of active y-faces
        to full modelling space (ie. nFy x nindActive_Fy )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-y projection matrix
        """
        if getattr(self, '_Pafy', None) is None:
            if self.indActive is None:
                self._Pafy = Utils.speye(self.mesh.nFy)
            else:
                # if getattr(self.mesh, 'aveCC2Fy', None) is not None:
                if self.mesh._meshType == "TREE":
                    if self.regularization_type == "Tikhonov":
                        indActive_Fy = (
                            (self.mesh.aveFy2CC.T * self.indActive) >= 1
                        )
                        self._Pafy = (
                            Utils.speye(self.mesh.nFy)[:, indActive_Fy]
                        )
                    else:
                        indActive_Fy = (
                            (
                                self.mesh._aveCC2FyStencil() *
                                self.indActive
                            ) >= 1
                        )
                        self._Pafy = (
                            Utils.speye(self.mesh.ntFy)[:, indActive_Fy]
                        )
                else:
                    indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) >= 1
                    self._Pafy = Utils.speye(self.mesh.nFy)[:, indActive_Fy]
        return self._Pafy

    @property
    def Pafz(self):
        """
        projection matrix that takes from the reduced space of active z-faces
        to full modelling space (ie. nFz x nindActive_Fz )

        :rtype: scipy.sparse.csr_matrix
        :return: active face-z projection matrix
        """
        if getattr(self, '_Pafz', None) is None:
            if self.indActive is None:
                self._Pafz = Utils.speye(self.mesh.nFz)
            else:
                # if getattr(self.mesh, 'aveCC2Fz', None) is not None:
                if self.mesh._meshType == "TREE":
                    if self.regularization_type == "Tikhonov":
                        indActive_Fz = (
                            (self.mesh.aveFz2CC.T * self.indActive) >= 1
                        )
                        self._Pafz = (
                            Utils.speye(self.mesh.nFz)[:, indActive_Fz]
                        )
                    else:
                        indActive_Fz = (
                            (
                                self.mesh._aveCC2FzStencil() *
                                self.indActive
                            ) >= 1
                        )
                        self._Pafz = (
                            Utils.speye(self.mesh.ntFz)[:, indActive_Fz]
                        )
                else:
                    indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) >= 1
                    self._Pafz = Utils.speye(self.mesh.nFz)[:, indActive_Fz]
        return self._Pafz

    @property
    def aveFx2CC(self):
        """
        averaging from active cell centers to active x-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active x-faces
        """
        if getattr(self, '_aveFx2CC', None) is None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveFx2CC = (
                        self.Pac.T * self.mesh.aveFx2CC * self.Pafx
                    )

                else:
                    nCinRow = mkvc((self.aveCC2Fx.T).sum(1))
                    nCinRow[nCinRow > 0] = 1./nCinRow[nCinRow > 0]
                    self._aveFx2CC = (
                        Utils.sdiag(nCinRow) *
                        self.aveCC2Fx.T
                    )

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
        if getattr(self, '_aveCC2Fx', None) is None:

            # if getattr(self.mesh, 'aveCC2Fx', None) is not None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveCC2Fx = (
                        Utils.sdiag(1./(self.aveFx2CC.T).sum(1)) *
                        self.aveFx2CC.T
                    )
                else:
                    self._aveCC2Fx = (
                        self.Pafx.T * self.mesh._aveCC2FxStencil() * self.Pac
                    )
            else:
                self._aveCC2Fx = (
                    Utils.sdiag(1./(self.aveFx2CC.T).sum(1)) * self.aveFx2CC.T
                )
        return self._aveCC2Fx

    @property
    def aveFy2CC(self):
        """
        averaging from active cell centers to active y-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active y-faces
        """
        if getattr(self, '_aveFy2CC', None) is None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveFy2CC = (
                        self.Pac.T * self.mesh.aveFy2CC * self.Pafy
                    )

                else:
                    nCinRow = mkvc((self.aveCC2Fy.T).sum(1))
                    nCinRow[nCinRow > 0] = 1./nCinRow[nCinRow > 0]
                    self._aveFy2CC = (
                        Utils.sdiag(nCinRow) *
                        self.aveCC2Fy.T
                    )

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
        if getattr(self, '_aveCC2Fy', None) is None:
            # if getattr(self.mesh, 'aveCC2Fy', None) is not None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveCC2Fy = (
                        Utils.sdiag(1./(self.aveFy2CC.T).sum(1)) *
                        self.aveFy2CC.T
                    )
                else:
                    self._aveCC2Fy = (
                        self.Pafy.T * self.mesh._aveCC2FyStencil() * self.Pac
                    )
            else:
                self._aveCC2Fy = (
                    Utils.sdiag(1./(self.aveFy2CC.T).sum(1)) * self.aveFy2CC.T
                )
        return self._aveCC2Fy

    @property
    def aveFz2CC(self):
        """
        averaging from active cell centers to active z-faces

        :rtype: scipy.sparse.csr_matrix
        :return: averaging from active cell centers to active z-faces
        """
        if getattr(self, '_aveFz2CC', None) is None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveFz2CC = (
                        self.Pac.T * self.mesh.aveFz2CC * self.Pafz
                    )

                else:
                    nCinRow = mkvc((self.aveCC2Fz.T).sum(1))
                    nCinRow[nCinRow > 0] = 1./nCinRow[nCinRow > 0]
                    self._aveFz2CC = (
                        Utils.sdiag(nCinRow) *
                        self.aveCC2Fz.T
                    )

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
        if getattr(self, '_aveCC2Fz', None) is None:
            # if getattr(self.mesh, 'aveCC2Fz', None) is not None:
            if self.mesh._meshType == "TREE":
                if self.regularization_type == "Tikhonov":
                    self._aveCC2Fz = (
                        Utils.sdiag(1./(self.aveFz2CC.T).sum(1)) *
                        self.aveFz2CC.T
                    )
                else:
                    self._aveCC2Fz = (
                        self.Pafz.T * self.mesh._aveCC2FzStencil() * self.Pac
                    )
            else:
                self._aveCC2Fz = (
                    Utils.sdiag(1./(self.aveFz2CC.T).sum(1)) * self.aveFz2CC.T
                )
        return self._aveCC2Fz

    @property
    def cellDiffx(self):
        """
        cell centered difference in the x-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the x-direction
        """
        if getattr(self, '_cellDiffx', None) is None:
            self._cellDiffx = self.Pafx.T * self.mesh.cellGradx * self.Pac
        return self._cellDiffx

    @property
    def cellDiffy(self):
        """
        cell centered difference in the y-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the y-direction
        """
        if getattr(self, '_cellDiffy', None) is None:
            self._cellDiffy = self.Pafy.T * self.mesh.cellGrady * self.Pac
        return self._cellDiffy

    @property
    def cellDiffz(self):
        """
        cell centered difference in the z-direction

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active cells in the z-direction
        """
        if getattr(self, '_cellDiffz', None) is None:
            self._cellDiffz = self.Pafz.T * self.mesh.cellGradz * self.Pac
        return self._cellDiffz

    @property
    def faceDiffx(self):
        """
        x-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the x-direction
        """
        if getattr(self, '_faceDiffx', None) is None:
            self._faceDiffx = self.Pac.T * self.mesh.faceDivx * self.Pafx
        return self._faceDiffx

    @property
    def faceDiffy(self):
        """
        y-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the y-direction
        """
        if getattr(self, '_faceDiffy', None) is None:
            self._faceDiffy = self.Pac.T * self.mesh.faceDivy * self.Pafy
        return self._faceDiffy

    @property
    def faceDiffz(self):
        """
        z-face differences

        :rtype: scipy.sparse.csr_matrix
        :return: differencing matrix for active faces in the z-direction
        """
        if getattr(self, '_faceDiffz', None) is None:
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
        if getattr(self, '_cellDiffxStencil', None) is None:

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
        if getattr(self, '_cellDiffyStencil', None) is None:

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
        if getattr(self, '_cellDiffzStencil', None) is None:

            self._cellDiffzStencil = (
                self.Pafz.T * self.mesh._cellGradzStencil * self.Pac
            )
        return self._cellDiffzStencil

