import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .base import BaseRegularization, BaseComboRegularization
from .. import utils


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

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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
            return utils.sdiag(np.sqrt(self.cell_weights))
        elif self._nC_residual != '*':
            return sp.eye(self._nC_residual)
        else:
            return utils.Identity()


class SimpleSmoothDeriv(BaseRegularization):
    """
    Base Simple Smooth Regularization. This base class regularizes on the first
    spatial derivative, not considering length scales, in the provided
    orientation

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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
        Weighting matrix that takes the first spatial difference
        with normalized length scales in the specified orientation
        """
        Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))
        W = utils.sdiag(self.length_scales) * getattr(
            self.regmesh,
            "cellDiff{orientation}Stencil".format(
                orientation=self.orientation
            )
        )
        if self.cell_weights is not None:

            W = (
                utils.sdiag(
                    (Ave*(self.cell_weights))**0.5
                ) * W
            )
        else:
            W = (
                utils.sdiag(
                    (Ave*self.regmesh.vol)**0.5
                ) * W
            )

        return W

    @property
    def length_scales(self):
        """
            Normalized cell based weighting

        """
        if getattr(self, '_length_scales', None) is None:
            Ave = getattr(self.regmesh, 'aveCC2F{}'.format(self.orientation))

            index = 'xyz'.index(self.orientation)

            length_scales = Ave * (
                self.regmesh.Pac.T*self.regmesh.mesh.h_gridded[:, index]
            )

            self._length_scales = length_scales.min() / length_scales

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

    - :math:`\phi_s` is a :class:`SimPEG.regularization.Small` instance
    - :math:`\phi_x` is a :class:`SimPEG.regularization.SimpleSmoothDeriv` instance, with :code:`orientation='x'`
    - :math:`\phi_y` is a :class:`SimPEG.regularization.SimpleSmoothDeriv` instance, with :code:`orientation='y'`
    - :math:`\phi_z` is a :class:`SimPEG.regularization.SimpleSmoothDeriv` instance, with :code:`orientation='z'`


    **Required Inputs**

    :param discretize.base.BaseMesh mesh: a SimPEG mesh

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

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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
            return utils.sdiag(np.sqrt(self.regmesh.vol * self.cell_weights))
        return utils.sdiag(np.sqrt(self.regmesh.vol))


class SmoothDeriv(BaseRegularization):
    """
    Base Smooth Regularization. This base class regularizes on the first
    spatial derivative in the provided orientation

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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
            self.mref = utils.Zero()

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

        return utils.sdiag(np.sqrt(Ave * vol)) * D


class SmoothDeriv2(BaseRegularization):
    """
    Base Smooth Regularization. This base class regularizes on the second
    spatial derivative in the provided orientation

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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
            utils.sdiag(vol**0.5) *
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

    :param discretize.base.BaseMesh mesh: SimPEG mesh
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

        self.regmesh.regularization_type = 'Tikhonov'
