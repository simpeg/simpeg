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

__all__ = [
    'Small',
    'SmoothDeriv', 'SmoothDeriv2',
    'SimpleSmoothDeriv',
    'Simple', 'Tikhonov',
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
                indActive_Fx = (self.mesh.aveFx2CC.T * self.indActive) == 1
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
                indActive_Fy = (self.mesh.aveFy2CC.T * self.indActive) == 1
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
                indActive_Fz = (self.mesh.aveFz2CC.T * self.indActive) == 1
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
                self.Pafx.T * self.mesh._cellGradxStencil() * self.Pac
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
                self.Pafy.T * self.mesh._cellGradyStencil() * self.Pac
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
                self.Pafz.T * self.mesh._cellGradzStencil() * self.Pac
            )
        return self._cellDiffzStencil


###############################################################################
#                                                                             #
#                          Single Regularization                              #
#                                                                             #
###############################################################################

class BaseRegularization(ObjectiveFunction.BaseObjectiveFunction):
    """
    Base class for regularization. Inherit this for building your own
    regularization. The base regularization assumes a weighted l2 style of
    regularization. However, if you wish to employ a different norm, the
    methods :meth:`__call__`, :meth:`deriv` and :meth:`deriv2` can be over-written

    **Optional Inputs**

    :param BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray mref: reference model
    :param numpy.ndarray indActive: active cell indices for reducing the size
    of differential operators in the definition of a regularization mesh

    """

    counter = None

    def __init__(
        self, mesh=None, **kwargs
    ):

        super(BaseRegularization, self).__init__()
        self._mesh = mesh
        Utils.setKwargs(self, **kwargs)

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

    # Observers and Validators
    @properties.validator('indActive')
    def _cast_to_bool(self, change):
        value = change['value']
        if value is not None:
            if value.dtype != 'bool':  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
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

    @properties.validator('mref')
    def _validate_mref(self, change):
        if not isinstance(change['value'], Utils.Zero) and self.nP != '*':
            assert len(change['value']) == self.nP, (
                'mref must be length {}'.format(self.nP)
            )

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
        if getattr(self.regmesh, 'nC', None) != '*':
            return self.regmesh.nC
        elif getattr(self, 'mapping', None) != '*':
            return self.mapping.shape[0]
        else:
            return self.nP

    @property
    def mesh(self):
        """
        a SimPEG mesh which the model is described on
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        assert isinstance(value, Mesh.BaseMesh) or value is None, (
            "mesh must be a SimPEG.Mesh object."
        )
        self._mesh = value

    @property
    def regmesh(self):
        """
        mesh used for creating operators for regularization. Excludes inactive
        cells if they are provided
        """
        if getattr(self, '_regmesh', None) is None:
            if self.indActive is not None:
                self._regmesh = RegularizationMesh(
                    self.mesh, indActive=self.indActive
                )
            else:
                self._regmesh = RegularizationMesh(self.mesh)
        return self._regmesh

    @regmesh.setter
    def regmesh(self, value):
        assert isinstance(value, RegularizationMesh) or value is None, (
            "regmesh must be an instance of a RegularizationMesh"
            )
        self._regmesh = value

    @property
    def mapping(self):
        """
        a mapping to map the model to the space in which you wish to regularize
        it in
        """
        if getattr(self, '_mapping', None) is None:
            return self.mapPair()
        return self._mapping

    @mapping.setter
    def mapping(self, value):
        if value is not None:
            value._assertMatchesPair(self.mapPair)
        self._mapping = value

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
            return Utils.sdiag(self.cell_weights)
        elif self._nC_residual != '*':
            return sp.eye(self._nC_residual)
        else:
            return Utils.Identity()


###############################################################################
#                                                                             #
#                           Combo Regularization                              #
#                                                                             #
###############################################################################

class BaseComboRegularization(ObjectiveFunction.ComboObjectiveFunction):

    mapPair = Maps.IdentityMap

    def __init__(
        self, mesh, objfcts=[],
        mapping=None, **kwargs
    ):

        self._mesh = mesh
        self._mapping = mapping

        super(BaseComboRegularization, self).__init__(
            objfcts=objfcts, multipliers=None
        )

        Utils.setKwargs(self, **kwargs)

    # Properties
    alpha_s = Props.Float("smallness weight")
    alpha_x = Props.Float("weight for the first x-derivative")
    alpha_y = Props.Float("weight for the first y-derivative")
    alpha_z = Props.Float("weight for the first z-derivative")
    alpha_xx = Props.Float("weight for the second x-derivative")
    alpha_yy = Props.Float("weight for the second y-derivative")
    alpha_zz = Props.Float("weight for the second z-derivative")

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

    @property
    def nP(self):
        if getattr(self.mapping, 'nP') != '*':
            return self.mapping.nP
        elif getattr(self.regmesh, 'nC') != '*':
            return self.regmesh.nC
        else:
            return '*'

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

    # Mirror property changes down to objective functions in objective function
    # list
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

    # Mirror other properties down

    @property
    def mesh(self):
        """
        a SimPEG mesh which the model is described on
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        assert isinstance(value, Mesh.BaseMesh) or value is None, (
            "mesh must be a SimPEG.Mesh object."
        )
        for fct in self.objfcts:
            fct.mesh = value
        self._mesh = value

    @property
    def regmesh(self):
        # This could be cleaned up
        if getattr(self, 'mesh', None) is not None:
            if getattr(self, '_regmesh', None) is None:
                self._regmesh = RegularizationMesh(mesh=self.mesh)
                if self.indActive is not None:
                    self._regmesh.indActive = self.indActive
            return self._regmesh
        return None

    @regmesh.setter
    def regmesh(self, val):
        for fct in self.objfcts:
            fct.regmesh = val
        self._regmesh = val

    @property
    def mapping(self):
        if getattr(self, '_mapping', None) is None:
            if getattr(self, 'regmesh', None) is not None:
                self._mapping = self.mapPair()
            else:
                self._mapping = None
        return self._mapping

    @mapping.setter
    def mapping(self, val):
        for fct in self.objfcts:
            fct.mapping = val
        self._mapping = val


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

        self.orientation = orientation
        assert self.orientation in ['x', 'y', 'z'], (
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
        W = getattr(
            self.regmesh,
            "cellDiff{orientation}Stencil".format(
                orientation=self.orientation
            )
        )
        if self.cell_weights is not None:
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))
            W = (
                Utils.sdiag(
                    (Ave*self.cell_weights)**0.5
                ) * W
            )
        return W


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
            Small(mesh=mesh, **kwargs),
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
        W = getattr(
            self.regmesh,
            "cellDiff{orientation}".format(
                orientation=self.orientation
            )
        )
        if self.cell_weights is not None:
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))
            W = (
                Utils.sdiag(
                    (Ave*self.cell_weights)**0.5
                ) * W
            )
        return W


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
        vol = self.regmesh.vol
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
        alpha_xx=Utils.Zero(), alpha_yy=Utils.Zero(), alpha_zz=Utils.Zero(),
        **kwargs
    ):

        objfcts = [
            Small(mesh=mesh, **kwargs),
            SmoothDeriv(mesh=mesh, orientation='x', **kwargs),
            SmoothDeriv2(mesh=mesh, orientation='x', **kwargs)
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

    norm = properties.Float(
        "norm used", default=2
    )

    space = properties.String(
        "By default inherit the objctive", default='linear'
    )

    scale = properties.Float(
        "General nob for scaling", default=1.
    )

    @property
    def stashedR(self):
        return self._stashedR

    @stashedR.setter
    def stashedR(self, value):
        self._stashedR = value

    @property
    def stashedR(self):
        return self._stashedR

    @stashedR.setter
    def stashedR(self, value):
        self._stashedR = value

    def R(self, f_m):
        # if R is stashed, return that instead
        if getattr(self, 'stashedR') is not None:
            return self.stashedR

        # Eta scaling is important for mix-norms...do not mess with it
        eta = (self.epsilon**(1.-self.norm/2.))**0.5
        r = eta / (f_m**2. + self.epsilon**2.)**((1.-self.norm/2.)/2.)

        self.stashedR = r  # stash on the first calculation
        return r


class SparseSmall(BaseSparse):
    """
    Sparse smallness regularization

    **Inputs**

    :param int norm: norm on the smallness
    """

    _multiplier_pair = 'alpha_s'

    def __init__(self, mesh, norm=2, **kwargs):
        super(SparseSmall, self).__init__(
            mesh=mesh, norm=norm, **kwargs
        )

    @property
    def f_m(self):
        return self.mapping * (self.model - self.mref)

    @property
    def W(self):
        if getattr(self, 'model', None) is None:
            R = Utils.speye(self.regmesh.nC)
        else:
            r = self.R(self.f_m) #, self.eps_p, self.norm)
            R = Utils.sdiag(r)

        if self.cell_weights is not None:
            return Utils.sdiag((self.scale * self.gamma *
                                self.cell_weights)**0.5) * R
        return (self.scale * self.gamma)**0.5 * R


class SparseDeriv(BaseSparse):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def __init__(self, mesh, orientation='x', **kwargs):

        self.orientation = orientation
        super(SparseDeriv, self).__init__(mesh=mesh, **kwargs)

    mrefInSmooth = properties.Bool(
        "include mref in the smoothness calculation?", default=False
    )

    @Utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
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
                        (self.scale * self.gamma * (Ave*(self.cell_weights)))**0.5
                    ) *
                    R
                )

            else:
                W = ((self.scale * self.gamma)**0.5) * R


            theta = self.cellDiffStencil * (self.mapping * m)
            dmdx = coterminal(theta)
            r = W * dmdx

        else:
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
                        (self.scale * self.gamma * (Ave*(self.cell_weights)))**0.5
                    ) *
                    R
                )

            else:
                W = ((self.scale * self.gamma)**0.5) * R

            theta = self.cellDiffStencil * (self.mapping * m)
            dmdx = coterminal(theta)

            r = W * dmdx

        else:
            r = self.W * (self.mapping * (self._delta_m(m)))
        mD = self.mapping.deriv(self._delta_m(m))
        return mD.T * (self.W.T * r)


    @property
    def _multiplier_pair(self):
        return 'alpha_{orientation}'.format(orientation=self.orientation)

    @property
    def f_m(self):
        if self.space == 'spherical':
            theta = self.cellDiffStencil * (self.mapping * self.model)
            dmdx = coterminal(theta)

        else:

            dmdx = self.cellDiffStencil * (self.mapping * self.model)

        return dmdx

    @property
    def cellDiffStencil(self):
        return getattr(
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

        if self.cell_weights is not None:
            return (
                Utils.sdiag(
                    (self.scale * self.gamma * (Ave*(self.cell_weights)))**0.5
                ) *
                R * self.cellDiffStencil
            )
        return ((self.scale * self.gamma)**0.5) * R * self.cellDiffStencil


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

        # Utils.setKwargs(self, **kwargs)

    # Properties
    norms = properties.Array(
        "Norms used to create the sparse regularization",
        default=[2., 2., 2., 2.]
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

    scale = properties.Float(
        "General nob for scaling", default=1.
    )
    # Save the l2 result during the IRLS
    l2model = None

    # Observers
    @properties.observer('norms')
    def _mirror_norms_to_objfcts(self, change):
        for i, objfct in enumerate(self.objfcts):
            objfct.norm = change['value'][i]

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

    @properties.observer('scale')
    def _mirror_scale_to_objfcts(self, change):
        for objfct in self.objfcts:
            objfct.scale = change['value']


def coterminal(theta):
    """ Compute coterminal angle so that [-pi < theta < pi]"""

    sub = theta[np.abs(theta) >= np.pi]
    sub = -np.sign(sub) * (2*np.pi-np.abs(sub))

    theta[np.abs(theta) >= np.pi] = sub

    return theta

