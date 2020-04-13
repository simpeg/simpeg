from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import warnings
import properties

from . import Utils
from . import Maps
from . import Mesh
from . import ObjectiveFunction
from . import Props
from .Utils import mkvc, speye, sdiag, kron3

__all__ = [
    'SimpleSmall', 'SimpleSmoothDeriv', 'Simple',
    'Small', 'SmoothDeriv', 'SmoothDeriv2', 'Tikhonov',
    'SparseSmall', 'SparseDeriv', 'Sparse',
]


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

    :param BaseMesh mesh: problem mesh
    :param numpy.array indActive: bool array, size nC, that is True where we have active cells. Used to reduce the operators so we regularize only on active cells

    """

    regularization_type = None # or 'Simple', 'Sparse' or 'Tikhonov'

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

        :rtype: numpy.array
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
                            (self.mesh._aveCC2FxStencil() * self.indActive) >= 1
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
                        print ("Use Tikhonov")
                        indActive_Fy = (
                            (self.mesh.aveFy2CC.T * self.indActive) >= 1
                        )
                        self._Pafy = (
                            Utils.speye(self.mesh.nFy)[:, indActive_Fy]
                        )
                    else:
                        indActive_Fy = (
                            (self.mesh._aveCC2FyStencil() * self.indActive) >= 1
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
                            (self.mesh._aveCC2FzStencil() * self.indActive) >= 1
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

###############################################################################
#                                                                             #
#                          Base Regularization                                #
#                                                                             #
###############################################################################


class BaseRegularization(ObjectiveFunction.BaseObjectiveFunction):
    """
    Base class for regularization. Inherit this for building your own
    regularization. The base regularization assumes a weighted l2 style of
    regularization. However, if you wish to employ a different norm, the
    methods :meth:`__call__`, :meth:`deriv` and :meth:`deriv2` can be
    over-written

    :param BaseMesh mesh: SimPEG mesh

    """

    def __init__(self, mesh=None, **kwargs):
        super(BaseRegularization, self).__init__()
        self.regmesh = RegularizationMesh(mesh)
        Utils.setKwargs(self, **kwargs)

    counter = None
    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=None
    )
    # Properties
    mref = Props.Array(
        "reference model"
    )
    indActive = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    cell_weights = properties.Array(
        "regularization weights applied at cell centers", dtype=float
    )
    regmesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        Maps.IdentityMap, default=Maps.IdentityMap()
    )

    # Observers and Validators
    @properties.validator('indActive')
    def _cast_to_bool(self, change):
        value = change['value']
        if value is not None:
            if value.dtype != 'bool':  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.regmesh.nC, dtype=bool)
                value[tmp] = True
                change['value'] = value

        # update regmesh indActive
        if getattr(self, 'regmesh', None) is not None:
            self.regmesh.indActive = Utils.mkvc(value)

    @properties.observer('indActive')
    def _update_regmesh_indActive(self, change):
        # update regmesh indActive
        if getattr(self, 'regmesh', None) is not None:
            self.regmesh.indActive = change['value']

    @properties.validator('cell_weights')
    def _validate_cell_weights(self, change):
        if change['value'] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != '*':
                assert len(change['value']) == self._nC_residual, (
                    'cell_weights must be length {} not {}'.format(
                        self._nC_residual, len(change['value'])
                    )
                )

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, 'nP') != '*':
            return self.mapping.nP
        elif getattr(self.regmesh, 'nC') != '*':
            return self.regmesh.nC
        else:
            return '*'

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        if getattr(self, 'mapping', None) != '*':
            return self.mapping.shape[0]
        elif getattr(self.regmesh, 'nC', None) != '*':
            return self.regmesh.nC
        else:
            return self.nP

    def _delta_m(self, m):
        if self.mref is None:
            return m
        return (-self.mref + m)  # in case self.mref is Zero, returns type m

    @Utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
        r = self.W * (self.mapping * (self._delta_m(m)))
        return 0.5 * r.dot(r)

    @Utils.timeIt
    def deriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """

        mD = self.mapping.deriv(self._delta_m(m))
        r = self.W * (self.mapping * (self._delta_m(m)))
        return mD.T * (self.W.T * r)

    @Utils.timeIt
    def deriv2(self, m, v=None):
        """
        Second derivative

        :param numpy.array m: geophysical model
        :param numpy.array v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
            W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W}

        """
        mD = self.mapping.deriv(self._delta_m(m))
        if v is None:
            return mD.T * self.W.T * self.W * mD

        return mD.T * ( self.W.T * ( self.W * ( mD * v) ) )


class Small(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model. Cell weights may be included.

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{W}^T
        \\mathbf{W} (\\mathbf{m} - \\mathbf{m_{ref}})

    where :math:`\\mathbf{m}` is the model, :math:`\\mathbf{m_{ref}}` is a
    reference model (default Zero) and :math:`\\mathbf{W}` is a weighting
    matrix (default Identity. If cell weights are provided, then it is
    :code:`diag(cell_weights)`)


    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh=None, **kwargs):

        super(Small, self).__init__(
            mesh=mesh, **kwargs
        )

    @property
    def W(self):
        """
        Weighting matrix
        """
        if self.cell_weights is not None:
            return Utils.sdiag((self.regmesh.vol*self.cell_weights)**0.5)
        # elif self._nC_residual != '*':
        #     return sp.eye(self._nC_residual)
        else:
            return Utils.sdiag(self.regmesh.vol**0.5)


###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################

class BaseComboRegularization(ObjectiveFunction.ComboObjectiveFunction):

    def __init__(
        self, mesh, objfcts=[], **kwargs
    ):

        super(BaseComboRegularization, self).__init__(
            objfcts=objfcts, multipliers=None
        )
        self.regmesh = RegularizationMesh(mesh)
        Utils.setKwargs(self, **kwargs)

        # link these attributes
        linkattrs = [
            'regmesh', 'indActive', 'cell_weights', 'mapping'
        ]

        for attr in linkattrs:
            val = getattr(self, attr)
            if val is not None:
                [setattr(fct, attr, val) for fct in self.objfcts]

    # Properties
    alpha_s = Props.Float("smallness weight")
    alpha_x = Props.Float("weight for the first x-derivative")
    alpha_y = Props.Float("weight for the first y-derivative")
    alpha_z = Props.Float("weight for the first z-derivative")
    alpha_xx = Props.Float("weight for the second x-derivative")
    alpha_yy = Props.Float("weight for the second y-derivative")
    alpha_zz = Props.Float("weight for the second z-derivative")

    counter = None

    mref = Props.Array(
        "reference model"
    )
    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )
    indActive = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    cell_weights = properties.Array(
        "regularization weights applied at cell centers", dtype=float
    )

    scale = properties.Float(
        "General nob for scaling", default=1.
    )

    regmesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        Maps.IdentityMap, default=Maps.IdentityMap()

    )

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, 'nP') != '*':
            return self.mapping.nP
        elif getattr(self.regmesh, 'nC') != '*':
            return self.regmesh.nC
        else:
            return '*'

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        if getattr(self.regmesh, 'nC', None) != '*':
            return self.regmesh.nC
        elif getattr(self, 'mapping', None) != '*':
            return self.mapping.shape[0]
        else:
            return self.nP

    def _delta_m(self, m):
        if self.mref is None:
            return m
        return (-self.mref + m)  # in case self.mref is Zero, returns type m

    @property
    def multipliers(self):
        """
        Factors that multiply the objective functions that are summed together
        to build to composite regularization
        """
        return [
            getattr(
                self, '{alpha}'.format(alpha=objfct._multiplier_pair)
            ) for objfct in self.objfcts
        ]

    # Observers and Validators
    @properties.validator('indActive')
    def _cast_to_bool(self, change):
        value = change['value']
        if value is not None:
            if value.dtype != 'bool':  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.regmesh.nC, dtype=bool)
                value[tmp] = True
                change['value'] = value

        # update regmesh indActive
        if getattr(self, 'regmesh', None) is not None:
            self.regmesh.indActive = Utils.mkvc(value)

    @properties.observer('indActive')
    def _update_regmesh_indActive(self, change):
        # update regmesh indActive
        if getattr(self, 'regmesh', None) is not None:
            self.regmesh.indActive = change['value']

    @properties.validator('cell_weights')
    def _validate_cell_weights(self, change):
        if change['value'] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != '*':
                assert len(change['value']) == self._nC_residual, (
                    'cell_weights must be length {} not {}'.format(
                        self._nC_residual, len(change['value'])
                    )
                )

    @properties.observer('mref')
    def _mirror_mref_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, 'mrefInSmooth', None) is not None:
                if self.mrefInSmooth is False:
                    fct.mref = Utils.Zero()
                else:
                    fct.mref = change['value']
            else:
                fct.mref = change['value']

    @properties.observer('mrefInSmooth')
    def _mirror_mrefInSmooth_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, 'mrefInSmooth', None) is not None:
                fct.mrefInSmooth = change['value']

    @properties.observer('indActive')
    def _mirror_indActive_to_objfctlist(self, change):
        value = change['value']
        if value is not None:
            if value.dtype != 'bool':
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
                value[tmp] = True
                change['value'] = value

        if getattr(self, 'regmesh', None) is not None:
            self.regmesh.indActive = value

        for fct in self.objfcts:
            fct.indActive = value

    @properties.observer('cell_weights')
    def _mirror_cell_weights_to_objfctlist(self, change):
        for fct in self.objfcts:
            fct.cell_weights = change['value']

    @properties.observer('mapping')
    def _mirror_mapping_to_objfctlist(self, change):
        for fct in self.objfcts:
            fct.mapping = change['value']


###############################################################################
#                                                                             #
#              Simple Regularization (no volume contribution)                 #
#                                                                             #
###############################################################################

class SimpleSmall(BaseRegularization):
    """
    Simple Small regularization - L2 regularization on the difference between a
    model and a reference model. Cell weights may be included. This does not
    include a volume contribution.

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{W}^T
        \\mathbf{W} (\\mathbf{m} - \\mathbf{m_{ref}})

    where :math:`\\mathbf{m}` is the model, :math:`\\mathbf{m_{ref}}` is a
    reference model and :math:`\\mathbf{W}` is a weighting
    matrix (default Identity). If cell weights are provided, then it is
    :code:`diag(np.sqrt(cell_weights))`)


    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh=None, **kwargs):

        super(SimpleSmall, self).__init__(
            mesh=mesh, **kwargs
        )

    @property
    def W(self):
        """
        Weighting matrix
        """
        if self.cell_weights is not None:
            return Utils.sdiag(np.sqrt(self.cell_weights))
        elif self._nC_residual != '*':
            return sp.eye(self._nC_residual)
        else:
            return Utils.Identity()


class SimpleSmall(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model. Cell weights may be included.

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{W}^T
        \\mathbf{W} (\\mathbf{m} - \\mathbf{m_{ref}})

    where :math:`\\mathbf{m}` is the model, :math:`\\mathbf{m_{ref}}` is a
    reference model (default Zero) and :math:`\\mathbf{W}` is a weighting
    matrix (default Identity. If cell weights are provided, then it is
    :code:`diag(cell_weights)`)


    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh=None, **kwargs):

        super(SimpleSmall, self).__init__(
            mesh=mesh, **kwargs
        )

    @property
    def W(self):
        """
        Weighting matrix
        """
        if self.cell_weights is not None:
            return Utils.sdiag(self.cell_weights**0.5)
        elif self._nC_residual != '*':
            return sp.eye(self._nC_residual)
        else:
            return Utils.Identity()


class SimpleSmoothDeriv(BaseRegularization):
    """
    Base Simple Smooth Regularization. This base class regularizes on the first
    spatial derivative, not considering length scales, in the provided
    orientation

    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool mrefInSmooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-mref) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
    """

    def __init__(
        self, mesh, orientation='x', **kwargs
    ):
        self.length_scales = None
        self.orientation = orientation
        assert self.orientation in ['x', 'y', 'z'], (
            "Orientation must be 'x', 'y' or 'z'"
        )

        self.gradientType = gradientType
        assert self.gradientType in ['orthogonal', 'total'], (
            "gradientType must be 'orthogonal' or 'total'"
        )

        if self.orientation == 'y':
            assert mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif self.orientation == 'z':
            assert mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )

        super(SimpleSmoothDeriv, self).__init__(
            mesh=mesh, **kwargs
        )

    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )

    @property
    def _multiplier_pair(self):
        return 'alpha_{orientation}'.format(orientation=self.orientation)

    @property
    def W(self):
        """
        Weighting matrix that takes the first spatial difference (no
        length scales considered) in the specified orientation
        """
        Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

        W = getattr(
            self.regmesh,
            "cellDiff{orientation}Stencil".format(
                orientation=self.orientation
            )
        )
        if self.cell_weights is not None:

            W = (
                Utils.sdiag(
                    (Ave*(self.cell_weights))**0.5
                ) * W
            )
        else:
            W = (
                Utils.sdiag(
                    (Ave*(self.regmesh.vol))**0.5
                ) * W
            )
        return W

    @property
    def length_scales(self):

        if getattr(self, '_length_scales', None) is None:
            index = 'xyz'.index(self.orientation)

            length_scales = (
                self.regmesh.Pac.T*self.regmesh.mesh.h_gridded[:, index]
            )**2.

            self._length_scales = length_scales / length_scales.min()

        return self._length_scales

    @length_scales.setter
    def length_scales(self, value):
        self._length_scales = value

class Simple(BaseComboRegularization):

    """
    Simple regularization that does not include length scales in the
    derivatives.

    .. math::

        r(\mathbf{m}) = \\alpha_s \phi_s + \\alpha_x \phi_x +
        \\alpha_y \phi_y + \\alpha_z \phi_z

    where:

    - :math:`\phi_s` is a :class:`SimPEG.Regularization.Small` instance
    - :math:`\phi_x` is a :class:`SimPEG.Regularization.SimpleSmoothDeriv` instance, with :code:`orientation='x'`
    - :math:`\phi_y` is a :class:`SimPEG.Regularization.SimpleSmoothDeriv` instance, with :code:`orientation='y'`
    - :math:`\phi_z` is a :class:`SimPEG.Regularization.SimpleSmoothDeriv` instance, with :code:`orientation='z'`


    **Required Inputs**

    :param BaseMesh mesh: a SimPEG mesh

    **Optional Inputs**

    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool mrefInSmooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-mref) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)

    **Weighting Parameters**

    :param float alpha_s: weighting on the smallness (default 1.)
    :param float alpha_x: weighting on the x-smoothness (default 1.)
    :param float alpha_y: weighting on the y-smoothness (default 1.)
    :param float alpha_z: weighting on the z-smoothness(default 1.)

    """

    def __init__(
        self, mesh,
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0,
        alpha_z=1.0, **kwargs
    ):

        objfcts = [
            SimpleSmall(mesh=mesh, **kwargs),
            SimpleSmoothDeriv(
                mesh=mesh, orientation='x',
                **kwargs
            )
        ]

        if mesh.dim > 1:
            objfcts.append(
                SimpleSmoothDeriv(
                    mesh=mesh, orientation='y',
                    **kwargs
                )
            )

        if mesh.dim > 2:
            objfcts.append(
                SimpleSmoothDeriv(
                    mesh=mesh, orientation='z',
                    **kwargs
                )
            )

        super(Simple, self).__init__(
            mesh=mesh, objfcts=objfcts, alpha_s=alpha_s, alpha_x=alpha_x,
            alpha_y=alpha_y, alpha_z=alpha_z, **kwargs
        )


###############################################################################
#                                                                             #
#         Tikhonov-Style Regularization (includes volume contribution)        #
#                                                                             #
###############################################################################

class Small(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model. Cell weights may be included. A volume
    contribution is included

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{W}^T
        \\mathbf{W} (\\mathbf{m} - \\mathbf{m_{ref}})

    where :math:`\\mathbf{m}` is the model, :math:`\\mathbf{m_{ref}}` is a
    reference model and :math:`\\mathbf{W}` is a weighting
    matrix (default :code:`diag(np.sqrt(vol))`. If cell weights are provided, then it is
    :code:`diag(np.sqrt(vol * cell_weights))`)


    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh=None, **kwargs):

        super(Small, self).__init__(
            mesh=mesh, **kwargs
        )

    @property
    def W(self):
        """
        Weighting matrix
        """
        if self.cell_weights is not None:
            return Utils.sdiag(np.sqrt(self.regmesh.vol * self.cell_weights))
        return Utils.sdiag(np.sqrt(self.regmesh.vol))


class SmoothDeriv(BaseRegularization):
    """
    Base Smooth Regularization. This base class regularizes on the first
    spatial derivative in the provided orientation

    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool mrefInSmooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-mref) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
    """

    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )

    def __init__(
        self, mesh, orientation='x', **kwargs
    ):

        self.orientation = orientation

        assert orientation in ['x', 'y', 'z'], (
                "Orientation must be 'x', 'y' or 'z'"
            )

        if self.orientation == 'y':
            assert mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif self.orientation == 'z':
            assert mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )

        super(SmoothDeriv, self).__init__(
            mesh=mesh, **kwargs
        )

        if self.mrefInSmooth is False:
            self.mref = Utils.Zero()

    @property
    def _multiplier_pair(self):
        return 'alpha_{orientation}'.format(orientation=self.orientation)

    @property
    def W(self):
        """
        Weighting matrix that constructs the first spatial derivative stencil
        in the specified orientation
        """

        vol = self.regmesh.vol.copy()
        if self.cell_weights is not None:
            vol *= self.cell_weights

        D = getattr(

            self.regmesh,
            "cellDiff{orientation}".format(
                orientation=self.orientation
            )
        )

        Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

        return Utils.sdiag(np.sqrt(Ave * vol)) * D


class SmoothDeriv2(BaseRegularization):
    """
    Base Smooth Regularization. This base class regularizes on the second
    spatial derivative in the provided orientation

    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool mrefInSmooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-mref) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
    """

    def __init__(
        self, mesh,
        orientation='x',
        **kwargs
    ):
        self.orientation = orientation

        if self.orientation == 'y':
            assert mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif self.orientation == 'z':
            assert mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )

        super(SmoothDeriv2, self).__init__(
            mesh=mesh, **kwargs
        )

    @property
    def _multiplier_pair(self):
        return 'alpha_{orientation}{orientation}'.format(
            orientation=self.orientation
        )

    @property
    def W(self):
        """
        Weighting matrix that takes the second spatial derivative in the
        specified orientation
        """
        vol = self.regmesh.vol.copy()
        if self.cell_weights is not None:
            vol *= self.cell_weights

        W = (
            Utils.sdiag(vol**0.5) *
            getattr(
                self.regmesh,
                'faceDiff{orientation}'.format(
                    orientation=self.orientation
                )
            ) *
            getattr(
                self.regmesh,
                'cellDiff{orientation}'.format(
                    orientation=self.orientation
                )
            )
        )
        return W


class Tikhonov(BaseComboRegularization):
    """
    L2 Tikhonov regularization with both smallness and smoothness (first order
    derivative) contributions.

    .. math::
        \phi_m(\mathbf{m}) = \\alpha_s \| W_s (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_x \| W_x \\frac{\partial}{\partial x} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_y \| W_y \\frac{\partial}{\partial y} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \\alpha_z \| W_z \\frac{\partial}{\partial z} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2

    Note if the key word argument `mrefInSmooth` is False, then mref is not
    included in the smoothness contribution.

    :param BaseMesh mesh: SimPEG mesh
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the thing you want to regularize
    :param numpy.ndarray indActive: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param bool mrefInSmooth: (default = False) put mref in the smoothness component?
    :param float alpha_s: (default 1e-6) smallness weight
    :param float alpha_x: (default 1) smoothness weight for first derivative in the x-direction
    :param float alpha_y: (default 1) smoothness weight for first derivative in the y-direction
    :param float alpha_z: (default 1) smoothness weight for first derivative in the z-direction
    :param float alpha_xx: (default 1) smoothness weight for second derivative in the x-direction
    :param float alpha_yy: (default 1) smoothness weight for second derivative in the y-direction
    :param float alpha_zz: (default 1) smoothness weight for second derivative in the z-direction
    """

    def __init__(
        self, mesh,
        alpha_s=1e-6, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        alpha_xx=0., alpha_yy=0., alpha_zz=0.,
        **kwargs
    ):

        objfcts = [
            Small(mesh=mesh, **kwargs),
            SmoothDeriv(mesh=mesh, orientation='x', **kwargs),
            # SmoothDeriv2(mesh=mesh, orientation='x', **kwargs)
        ]

        if mesh.dim > 1:
            objfcts += [
                SmoothDeriv(mesh=mesh, orientation='y', **kwargs),
                SmoothDeriv2(mesh=mesh, orientation='y', **kwargs)
            ]

        if mesh.dim > 2:
            objfcts += [
                SmoothDeriv(mesh=mesh, orientation='z', **kwargs),
                SmoothDeriv2(mesh=mesh, orientation='z', **kwargs)
            ]

        super(Tikhonov, self).__init__(
            mesh,
            alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
            alpha_xx=alpha_xx, alpha_yy=alpha_yy, alpha_zz=alpha_zz,
            objfcts=objfcts, **kwargs
        )

        self.regmesh.regularization_type = 'Tikhonov'


class BaseSparse(BaseRegularization):
    """
    Base class for building up the components of the Sparse Regularization
    """



    def __init__(self, mesh, **kwargs):
        self._stashedR = None
        super(BaseSparse, self).__init__(mesh=mesh, **kwargs)

    model = properties.Array(
        "current model", dtype=float
    )

    gamma = properties.Float(
        "Model norm scaling to smooth out convergence", default=1.
    )

    epsilon = properties.Float(
        "Threshold value for the model norm", default=1e-3,
        required=True
    )

    norm = properties.Array(
        "norm used", dtype=float
    )

    space = properties.String(
        "By default inherit the objctive", default='linear'
    )

    gradientType = properties.String(
        "type of gradient", default='total'
    )

    scale = properties.Array(
        "General nob for scaling", dtype=float
    )

    # Give the option to scale or not
    scaledIRLS = properties.Bool(
        "Scale the gradients of the IRLS norms",
        default=True
    )

    @properties.validator('scale')
    def _validate_scale(self, change):
        if change['value'] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != '*':
                assert len(change['value']) == self._nC_residual, (
                    'scale must be length {} not {}'.format(
                        self._nC_residual, len(change['value'])
                    )
                )
    @property
    def stashedR(self):
        return self._stashedR

    @stashedR.setter
    def stashedR(self, value):
        self._stashedR = value


class SparseSmall(BaseSparse):
    """
    Sparse smallness regularization

    **Inputs**

    :param int norm: norm on the smallness
    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh, **kwargs):
        super(SparseSmall, self).__init__(
            mesh=mesh, **kwargs
        )

    # Give the option to scale or not
    scaledIRLS = properties.Bool(
        "Scale the gradients of the IRLS norms",
        default=True
    )

    @property
    def f_m(self):

        return self.mapping * self._delta_m(self.model)

    @property
    def W(self):
        if getattr(self, 'model', None) is None:
            R = Utils.speye(self.mapping.shape[0])
        else:
            r = self.R(self.f_m)
            R = Utils.sdiag(r)

        if self.scale is None:

            self.scale = np.ones(self.mapping.shape[0])

        if self.cell_weights is not None:
            return Utils.sdiag((self.scale *
                                self.cell_weights)**0.5) * R
        return Utils.sdiag((self.scale*self.regmesh.vol)**0.5) * R

    def R(self, f_m):
        # if R is stashed, return that instead
        if getattr(self, 'stashedR') is not None:
            return self.stashedR


        # Default to 1 for zero gradients
        eta = np.ones_like(f_m)

        if self.scaledIRLS:
            # Eta scaling is important for mix-norms...do not mess with it
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()
            maxVal[self.norm < 1] = self.epsilon / np.sqrt(1.-self.norm[self.norm < 1])
            maxGrad = maxVal / (maxVal**2. + self.epsilon**2.)**(1.-self.norm/2.)

            eta[maxGrad != 0] = np.abs(f_m).max()/maxGrad[maxGrad != 0]

        r = (eta / (f_m**2. + self.epsilon**2.)**(1.-self.norm/2.))**0.5

        # Theoritical full deriv for testing
        # r = (
        #     eta * (
        #         1. / (f_m**2. + self.epsilon**2.)**(1.-self.norm/2.) +
        #         (self.norm/2. - 1.) * f_m**2. / (f_m**2. + self.epsilon**2.)**(2.-self.norm/2.)
        #     )

        # )**0.5
        self.stashedR = r  # stash on the first calculation
        return r

    @Utils.timeIt
    def deriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """

        mD = self.mapping.deriv(self._delta_m(m))
        r = self.gamma * self.W * (self.mapping * (self._delta_m(m)))
        return mD.T * (self.W.T * r)


class SparseDeriv(BaseSparse):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def __init__(self, mesh, orientation='x', **kwargs):

        self.length_scales = None
        self.orientation = orientation
        super(SparseDeriv, self).__init__(mesh=mesh, **kwargs)

    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )

    # Give the option to scale or not
    scaledIRLS = properties.Bool(
        "Scale the gradients of the IRLS norms",
        default=True
    )

    @Utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """

        if self.mrefInSmooth:

            f_m = self._delta_m(m)

        else:
            f_m = m

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])
        if self.space == 'spherical':
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

            if getattr(self, 'model', None) is None:
                R = Utils.speye(self.cellDiffStencil.shape[0])

            else:

                r = self.R(self.f_m)
                R = Utils.sdiag(r)

            if self.cell_weights is not None:
                W = (
                    Utils.sdiag(
                        (Ave*(self.scale * self.cell_weights))**0.5
                    ) *
                    R
                )

            else:
                W = Utils.sdiag((Ave * (self.scale*self.regmesh.vol))**0.5) * R


            dmdx = self.cellDiffStencil * (self.mapping * f_m)
            dmdx = coterminal(dmdx)
            r = W * dmdx

        else:
            r = self.W * (self.mapping * f_m)

        return 0.5*r.dot(r)

    def R(self, f_m):
        # if R is stashed, return that instead
        if getattr(self, 'stashedR') is not None:
            return self.stashedR

        # Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))


        eta = np.ones_like(f_m)

        if self.scaledIRLS:
            # Eta scaling is important for mix-norms...do not mess with it
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()
            maxVal[self.norm < 1] = self.epsilon / np.sqrt(1.-self.norm[self.norm < 1])
            maxGrad = maxVal / (maxVal**2. + (self.epsilon)**2.)**(1.-self.norm/2.)

            eta[maxGrad != 0] = np.abs(f_m).max()/maxGrad[maxGrad != 0]

        r = (eta / (f_m**2. + (self.epsilon)**2.)**(1.-self.norm/2.))**0.5

        # Theoritical full deriv for testing
        # r = (
        #     eta * (
        #         1. / (f_m**2. + self.epsilon**2.)**(1.-self.norm/2.) +
        #         (self.norm/2. - 1.) * f_m**2. / (f_m**2. + self.epsilon**2.)**(2.-self.norm/2.)
        #     )

        # )**0.5

        # print(eta)
        self.stashedR = r  # stash on the first calculation
        return r

    @Utils.timeIt
    def deriv(self, m):
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

        """

        if self.mrefInSmooth:

            f_m = self._delta_m(m)

        else:
            f_m = m

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])


        if self.space == 'spherical':
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

            if getattr(self, 'model', None) is None:
                R = Utils.speye(self.cellDiffStencil.shape[0])

            else:
                r = self.R(self.f_m)
                R = Utils.sdiag(r)

            if self.cell_weights is not None:
                W = (
                    Utils.sdiag(
                        ((Ave * (self.scale * self.cell_weights)))**0.5
                    ) *
                    R
                )

            else:
                W = Utils.sdiag((Ave * (self.scale*self.regmesh.vol))**0.5) * R


            dmdx = self.cellDiffStencil * (self.mapping * f_m)
            dmdx = coterminal(dmdx)

            r = self.gamma * W * dmdx

        else:
            r = self.gamma * self.W * (self.mapping * f_m)

        mD = self.mapping.deriv(f_m)
        return mD.T * (self.W.T * r)

    @property
    def _multiplier_pair(self):
        return 'alpha_{orientation}'.format(orientation=self.orientation)

    @property
    def f_m(self):

        if self.mrefInSmooth:

            f_m = self._delta_m(self.model)

        else:
            f_m = self.model

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        if self.gradientType == 'total':
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

            dmdx = np.abs(self.regmesh.aveFx2CC *
                          self.regmesh.cellDiffxStencil *
                          (self.mapping * f_m)
                          )

            if self.space == 'spherical':
                # theta = self.cellDiffStencil * (self.mapping * f_m)
                dmdx = coterminal(dmdx)

            if self.regmesh.dim > 1:

                dmdy = np.abs(
                    self.regmesh.aveFy2CC *
                    self.regmesh.cellDiffyStencil *
                    (self.mapping * f_m)
                )

                if self.space == 'spherical':
                    # theta = self.cellDiffStencil * (self.mapping * f_m)
                    dmdx += coterminal(dmdy)
                else:
                    dmdx += dmdy

            if self.regmesh.dim > 2:

                dmdz = np.abs(
                    self.regmesh.aveFz2CC *
                    self.regmesh.cellDiffzStencil *
                    (self.mapping * f_m)
                )

                if self.space == 'spherical':
                    # theta = self.cellDiffStencil * (self.mapping * f_m)
                    dmdx += coterminal(dmdz)
                else:
                    dmdx += dmdz

            dmdx = Ave * dmdx

            if self.space == 'spherical':
                # theta = self.cellDiffStencil * (self.mapping * f_m)
                dmdx = coterminal(dmdx)

        else:
            dmdx = self.cellDiffStencil * (self.mapping * f_m)

        return dmdx

    @property
    def cellDiffStencil(self):

        return Utils.sdiag(self.length_scales) * getattr(
            self.regmesh, 'cellDiff{}Stencil'.format(self.orientation)
        )

    @property
    def W(self):

        Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

        if getattr(self, 'model', None) is None:
            R = Utils.speye(self.cellDiffStencil.shape[0])

        else:
            r = self.R(self.f_m)
            R = Utils.sdiag(r)

        if self.scale is None:
            self.scale = np.ones(self.mapping.shape[0])

        if self.cell_weights is not None:
            return (
                Utils.sdiag(
                    (Ave*(self.scale * self.cell_weights))**0.5
                ) *
                R * self.cellDiffStencil
            )
        return Utils.sdiag((Ave*(self.scale*self.regmesh.vol))**0.5) * R * self.cellDiffStencil

    @property
    def length_scales(self):

        if getattr(self, '_length_scales', None) is None:
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

            index = 'xyz'.index(self.orientation)

            length_scales = Ave * (
                (self.regmesh.Pac.T*self.regmesh.mesh.h_gridded[:, index])
            )

            self._length_scales = length_scales.min()/length_scales

        return self._length_scales

    @length_scales.setter
    def length_scales(self, value):
        self._length_scales = value

class Sparse(BaseComboRegularization):
    """
    The regularization is:

    .. math::

        R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R
        W(m-m_\\text{ref})}

    where the IRLS weight

    .. math::

        R = \eta TO FINISH LATER!!!

    So the derivative is straight forward:

    .. math::

        R(m) = \mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

    The IRLS weights are recomputed after each beta solves.
    It is strongly recommended to do a few Gauss-Newton iterations
    before updating.
    """
    def __init__(
        self, mesh,
        alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0,
        **kwargs
    ):

        objfcts = [
            SparseSmall(mesh=mesh, **kwargs),
            SparseDeriv(mesh=mesh, orientation='x', **kwargs)
        ]

        if mesh.dim > 1:
            objfcts.append(SparseDeriv(mesh=mesh, orientation='y', **kwargs))

        if mesh.dim > 2:
            objfcts.append(SparseDeriv(mesh=mesh, orientation='z', **kwargs))

        super(Sparse, self).__init__(
            mesh=mesh, objfcts=objfcts,
            alpha_s=alpha_s, alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z,
            **kwargs
        )
        self.mesh = mesh
        # Utils.setKwargs(self, **kwargs)

    # Properties
    norms = properties.Array(
        "Norms used to create the sparse regularization",
        default=np.c_[2., 2., 2., 2.], shape={('*', '*')}
    )

    eps_p = properties.Float(
        "Threshold value for the model norm", required=True
        )

    eps_q = properties.Float(
        "Threshold value for the model gradient norm", required=True
        )

    model = properties.Array("current model", dtype=float)

    gamma = properties.Float(
        "Model norm scaling to smooth out convergence", default=1.
    )

    space = properties.String(
        "type of model", default='linear'
    )

    gradientType = properties.String(
        "type of gradient", default='components'
    )

    scales = properties.Array(
        "General nob for scaling",
        default=np.c_[1., 1., 1., 1.], shape={('*', '*')}
    )

    # Give the option to scale or not
    scaledIRLS = properties.Bool(
        "Scale the gradients of the IRLS norms",
        default=True
    )

    # Save the l2 result during the IRLS
    l2model = None

    @properties.validator('norms')
    def _validate_norms(self, change):
        if change['value'].shape[0] == 1:
            change['value'] = np.kron(np.ones((self.regmesh.Pac.shape[1], 1)), change['value'])
        elif change['value'].shape[0] > 1:
            assert change['value'].shape[0] == self.regmesh.Pac.shape[1], (
                "Vector of norms must be the size of active model parameters ({})"
                "The provided vector has length "
                "{}".format(
                    self.regmesh.Pac.shape[0], len(change['value'])
                )
            )

    # Observers
    @properties.observer('norms')
    def _mirror_norms_to_objfcts(self, change):

        self.objfcts[0].norm = change['value'][:, 0]
        for i, objfct in enumerate(self.objfcts[1:]):
            Ave = getattr(objfct.regmesh, 'aveCC2F{}'.format(objfct.orientation))
            objfct.norm = Ave*change['value'][:,i+1]

    @properties.observer('model')
    def _mirror_model_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.model = change['value']

    @properties.observer('gamma')
    def _mirror_gamma_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.gamma = change['value']

    @properties.observer('eps_p')
    def _mirror_eps_p_to_smallness(self, change):
        for objfct in self.objfcts:
            if isinstance(objfct, SparseSmall):
                objfct.epsilon = change['value']

    @properties.observer('eps_q')
    def _mirror_eps_q_to_derivs(self, change):
        for objfct in self.objfcts:
            if isinstance(objfct, SparseDeriv):
                objfct.epsilon = change['value']

    @properties.observer('space')
    def _mirror_space_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.space = change['value']

    @properties.observer('scaledIRLS')
    def _mirror_scaledIRLS_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.scaledIRLS = change['value']

    @properties.observer('gradientType')
    def _mirror_gradientType_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.gradientType = change['value']

    @properties.validator('scales')
    def _validate_scales(self, change):
        if change['value'].shape[0] == 1:
            change['value'] = np.kron(np.ones((self.regmesh.Pac.shape[1], 1)), change['value'])
        elif change['value'].shape[0] > 1:
            assert change['value'].shape[0] == self.regmesh.Pac.shape[1], (
                "Vector of scales must be the size of active model parameters ({})"
                "The provided vector has length "
                "{}".format(
                    self.regmesh.Pac.shape[0], len(change['value'])
                )
            )

    # Observers
    @properties.observer('scales')
    def _mirror_scale_to_objfcts(self, change):
        for i, objfct in enumerate(self.objfcts):
            objfct.scale = change['value'][:,i]


def coterminal(theta):
    """ Compute coterminal angle so that [-pi < theta < pi]"""
    sub = theta[np.abs(theta) >= np.pi]
    sub = -np.sign(sub) * (2*np.pi-np.abs(sub))

    theta[np.abs(theta) >= np.pi] = sub

    return theta


def ddx(n, vals):
    """Define 1D averaging operator from cell-centers to nodes."""
    ddx = (
        sp.spdiags(
            (np.ones((n, 1)) * vals).T,
            [-1, 0, 1],
            n, n,
            format="csr"
        )
    )
    return ddx


def getDiffOpRot(mesh, psi, theta, phi, vec, forward=True):

    import scipy as sp
    assert mesh.dim > 1, 'Only for mesh 2D and 3D'

    def getCellNeighbors(mesh):
        Dx = mesh._cellGradxStencil
        Dy = mesh._cellGradyStencil
        # Get the current IJ of the stencil derive
        Ix, Jx, _ = sp.sparse.find(Dx)
        Iy, Jy, _ = sp.sparse.find(Dy)
        jx = np.sort(Jx[np.argsort(Ix)].reshape((int(Ix.shape[0]/2), 2)), axis=1)
        jy = np.sort(Jy[np.argsort(Iy)].reshape((int(Iy.shape[0]/2), 2)), axis=1)
        jx_bck = np.c_[jx[:, 1], jx[:, 0]]
        jy_bck = np.c_[jy[:, 1], jy[:, 0]]

        maxInd = np.max([jx.max(), jy.max()])

        if mesh.dim == 3:
            Dz = mesh._cellGradzStencil
            Iz, Jz, _ = sp.sparse.find(Dz)
            jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0]/2), 2)), axis=1)
            jz_bck = np.c_[jz[:, 1], jz[:, 0]]

            maxInd = np.max([jz.max(), maxInd])


        # Cycle through the gradients forward and backward to deal with multiple
        # levels on Tree mesh
        # Pre-allocate index arrays
        jAll = []  # Store
        div_xy = np.ones(maxInd+1, dtype='int')*-1
        div_yx = np.ones(maxInd+1, dtype='int')*-1
        div_xyb = np.ones(maxInd+1, dtype='int')*-1
        div_yxb = np.ones(maxInd+1, dtype='int')*-1

        div_xy[jy[:, 0]] = jy[:, 1]  # Find y neigbour of x adjacent
        div_yx[jx[:, 1]] = jx[:, 0]  # Find x neigbour of y adjacent

        div_xyb[jy_bck[:, 0]] = jy_bck[:, 1]  # Find y neigbour of x adjacent backward
        div_yxb[jx_bck[:, 1]] = jx_bck[:, 0]  # Find x neigbour of y adjacent backward

        jAll += [jx]
        jAll += [jy]

        jAll += [np.c_[jx[:, 0], div_xy[jx[:, 1]]]]
        jAll += [np.c_[jx[:, 1], div_xy[jx[:, 0]]]]

        jAll += [np.c_[div_yx[jy[:, 0]], jy[:, 1]]]
        jAll += [np.c_[div_yx[jy[:, 1]], jy[:, 0]]]

        # Repeat backward for Treemesh
        jAll += [jx_bck]
        jAll += [jy_bck]

        jAll += [np.c_[jx_bck[:, 0], div_xyb[jx_bck[:, 1]]]]
        jAll += [np.c_[jx_bck[:, 1], div_xyb[jx_bck[:, 0]]]]

        # Stack all and keep only unique pairs
        jAll = np.vstack(jAll)
        jAll = np.unique(jAll, axis=0)

        # Remove all the -1 for TreeMesh
        jAll = jAll[(jAll[:, 0] != -1) & (jAll[:, 1] != -1), :]

        # Use all the neighbours on the xy plane to find neighbours in z
        if mesh.dim == 3:
            jAllz = []
            div_z = np.ones(maxInd+1, dtype='int')*-1
            div_zb = np.ones(maxInd+1, dtype='int')*-1

            div_z[jz[:, 0]] = jz[:, 1]
            div_zb[jz_bck[:, 0]] = jz_bck[:, 1]

            jAllz += [jz]
            jAllz += [jz_bck]

            jAllz += [np.c_[jAll[:, 0], div_z[jAll[:, 1]]]]
            jAllz += [np.c_[jAll[:, 1], div_z[jAll[:, 0]]]]

            jAllz += [np.c_[jAll[:, 0], div_zb[jAll[:, 1]]]]
            jAllz += [np.c_[jAll[:, 1], div_zb[jAll[:, 0]]]]

            # Stack all and keep only unique pairs
            jAll = np.vstack([jAll, np.vstack(jAllz)])
            jAll = np.unique(jAll, axis=0)

            # Remove all the -1 for TreeMesh
            jAll = jAll[(jAll[:, 0] != -1) & (jAll[:, 1] != -1), :]

        return jAll

    hx = mesh.h_gridded[:, 0]
    hy = mesh.h_gridded[:, 1]

    if isinstance(phi, float):
        phi = np.ones(mesh.nC) * phi
    phi = np.arctan2((np.sin(phi)/hy), (np.cos(phi)/hx))

    if mesh.dim == 3:
        hz = mesh.h_gridded[:, 2]

        if isinstance(theta, float):
            theta = np.ones(mesh.nC) * theta
        theta = np.arctan2((np.sin(theta)/hz), (np.cos(theta)/hx))

        if isinstance(psi, float):
            psi = np.ones(mesh.nC) * psi
        psi = np.arctan2((np.sin(psi)/hz), (np.cos(psi)/hy))

    if forward:
        ind = 1
    else:
        ind = -1

    if mesh.dim == 2:
        if vec == 'X':
            px = np.kron(np.ones(mesh.nC), np.c_[ind, 0])

        elif vec == 'Y':
            px = np.kron(np.ones(mesh.nC), np.c_[0, ind])

    if mesh.dim == 3:

        if vec == 'X':
            px = np.kron(np.ones(mesh.nC), np.c_[ind, 0, 0])
        elif vec == 'Y':
            px = np.kron(np.ones(mesh.nC), np.c_[0, ind, 0])
        else:
            px = np.kron(np.ones(mesh.nC), np.c_[0, 0, ind])

    if mesh.dim == 2:
        rza = mkvc(np.c_[np.cos(phi), np.cos(phi)].T)
        rzb = mkvc(np.c_[np.sin(phi), np.zeros(mesh.nC)].T)
        rzc = mkvc(np.c_[-np.sin(phi), np.zeros(mesh.nC)].T)
        Rz = sp.sparse.diags([rzb[:-1], rza, rzc[:-1]], [-1, 0, 1])
        rx = (Rz*px.T).reshape((mesh.nC, 2))
    else:
        # Create sparse rotation operators
        rxa = mkvc(np.c_[np.ones(mesh.nC), np.cos(psi), np.cos(psi)].T)
        rxb = mkvc(np.c_[np.zeros(mesh.nC), np.sin(psi), np.zeros(mesh.nC)].T)
        rxc = mkvc(np.c_[np.zeros(mesh.nC), -np.sin(psi), np.zeros(mesh.nC)].T)
        Rx = sp.sparse.diags([rxb[:-1], rxa, rxc[:-1]], [-1, 0, 1])

        rya = mkvc(np.c_[np.cos(theta), np.ones(mesh.nC), np.cos(theta)].T)
        ryb = mkvc(np.c_[np.sin(theta), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
        ryc = mkvc(np.c_[-np.sin(theta), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
        Ry = sp.sparse.diags([ryb[:-2], rya, ryc[:-2]], [-2, 0, 2])

        rza = mkvc(np.c_[np.cos(phi), np.cos(phi), np.ones(mesh.nC)].T)
        rzb = mkvc(np.c_[np.sin(phi), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
        rzc = mkvc(np.c_[-np.sin(phi), np.zeros(mesh.nC), np.zeros(mesh.nC)].T)
        Rz = sp.sparse.diags([rzb[:-1], rza, rzc[:-1]], [-1, 0, 1])

        # Rotate all cell vectors
        rx = (Rz*(Ry*(Rx*px.T))).reshape((mesh.nC, 3))

    jd = getCellNeighbors(mesh)
    # Move the bottom-SW and top-NE nodes of stencil cell
    nBSW = mesh.gridCC[jd[:, 0], :] - mesh.h_gridded[jd[:, 0], :]/2 + rx[jd[:, 0], :]*mesh.h_gridded[jd[:, 0], :]
    nTNE = mesh.gridCC[jd[:, 0], :] + mesh.h_gridded[jd[:, 0], :]/2 + rx[jd[:, 0], :]*mesh.h_gridded[jd[:, 0], :]

    # Get corners for neighbours
    sBSW = mesh.gridCC[jd[:, 1], :] - mesh.h_gridded[jd[:, 1], :]/2
    sTNE = mesh.gridCC[jd[:, 1], :] + mesh.h_gridded[jd[:, 1], :]/2

    # Compute fractional volumes with base stencil
    V = (
            np.max([
                np.min([sTNE[:, 0], nTNE[:, 0]], axis=0) -
                np.max([sBSW[:, 0], nBSW[:, 0]], axis=0),
                np.zeros(jd.shape[0])], axis=0) *
            np.max([
                np.min([sTNE[:, 1], nTNE[:, 1]], axis=0) -
                np.max([sBSW[:, 1], nBSW[:, 1]], axis=0),
                np.zeros(jd.shape[0])], axis=0))

    if mesh.dim == 3:
        V *= np.max([
                np.min([sTNE[:, 2], nTNE[:, 2]], axis=0) -
                np.max([sBSW[:, 2], nBSW[:, 2]], axis=0),
                np.zeros(jd.shape[0])], axis=0)

    # Remove all rows of zero
    ind = (V > 0) * (jd[:, 0] != jd[:, 1])
    jd = jd[ind, :]
    V = V[ind]

    Dx2 = sp.sparse.csr_matrix((V, (jd[:, 0], jd[:, 1])), shape=(mesh.nC, mesh.nC))

    # Normalize rows
    V = mkvc(sp.sum(Dx2, axis=1))
    V[V > 0] = 1. / V[V > 0]
    Dx2 = -sdiag(V) * Dx2

    diag = np.ones(mesh.nC)
    diag[V == 0] = 0

    Dx = (sdiag(diag) + Dx2)
    return Dx
