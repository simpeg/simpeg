from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import warnings
import properties

from .. import props
from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import RegularizationMesh


class BaseRegularization(BaseObjectiveFunction):
    """
    Base class for regularization. Inherit this for building your own
    regularization. The base regularization assumes a weighted l2 style of
    regularization. However, if you wish to employ a different norm, the
    methods :meth:`__call__`, :meth:`deriv` and :meth:`deriv2` can be
    over-written

    :param discretize.base.BaseMesh mesh: SimPEG mesh

    """
    _active_cells = None
    _cell_weights = None
    _mapping = None
    _reference_model = None
    _regularization_mesh = None

    def __init__(self, mesh=None, **kwargs):
        super().__init__()
        self.regularization_mesh = RegularizationMesh(mesh)
        if "active_cells" in kwargs.keys():
            self.regularization_mesh.active_cells = kwargs.pop("active_cells")
        utils.setKwargs(self, **kwargs)

    counter = None

    # Properties
    @property
    def reference_model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        self.validate_array_type("reference_model", values, float)
        self.validate_shape("reference_model", values, self._nC_residual)

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.reference_model = utils.mkvc(values)

        self._reference_model = values

    @property
    def active_cells(self) -> np.ndarray:
        """Indices of active cells in the mesh"""
        return self._active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        self.validate_array_type("active_cells", values, bool)
        self.validate_shape("active_cells", values, self._nC_residual)

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = utils.mkvc(values)

        self._active_cells = values

    @property
    def cell_weights(self):
        """Regularization weights applied at cell centers"""
        return self._cell_weights

    @cell_weights.setter
    def cell_weights(self, values: np.ndarray):
        self.validate_array_type("cell_weights", values, float)
        self.validate_shape("cell_weights", values)
        self._cell_weights = values

    @property
    def regularization_mesh(self):
        return self._regularization_mesh
        properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )

    @property
    def mapping(self):
        return self._mapping
        properties.Instance(
        "mapping which is applied to model in the regularization",
        maps.IdentityMap,
        default=maps.IdentityMap(),
    )

    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, "nP") != "*":
            return self.mapping.nP
        elif getattr(self.regularization_mesh, "nC") != "*":
            return self.regularization_mesh.nC
        else:
            return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """

        nC = getattr(self.regularization_mesh, "nC", None)
        mapping = getattr(self, "mapping", None)

        if nC != "*" and nC is not None:
            return self.regularization_mesh.nC
        elif mapping is not None and mapping.shape[0] != "*":
            return self.mapping.shape[0]
        else:
            return self.nP

    def _delta_m(self, m):
        if self.reference_model is None:
            return m
        return -self.reference_model + m  # in case self.reference_model is Zero, returns type m

    @utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
        r = self.W * (self.mapping * (self._delta_m(m)))
        return 0.5 * r.dot(r)

    @utils.timeIt
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

    @utils.timeIt
    def deriv2(self, m, v=None):
        """
        Second derivative

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
            W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \\mathbf{W^\\top W}

        """

        mD = self.mapping.deriv(self._delta_m(m))
        if v is None:
            return mD.T * self.W.T * self.W * mD

        return mD.T * (self.W.T * (self.W * (mD * v)))

    @staticmethod
    def validate_array_type(attribute, array, dtype):
        """Generic array and type validator"""
        if not isinstance(array, np.ndarray) and not array.dtype == dtype:
            TypeError(
                f"{attribute} must by a {np.ndarray} of type {dtype}. "
                f"Values of type {type(array)} was provided."
            )

    @staticmethod
    def validate_shape(attribute, values, shape):
        """Generic array shape validator"""
        if (
            shape != "*"
            and len(values) != shape
        ):
            raise ValueError(
                f"{attribute} must be length {shape} not {len(values)}"
            )


class Small(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model. Cell weights normalized by cell volumes
    may be included.

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
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

    @property
    def W(self):
        """
        Weighting matrix
        """

        weights = self.scale * self.regularization_mesh.vol

        if self.cell_weights is not None:
            weights *= self.cell_weights

        return utils.sdiag(weights ** 0.5)


class SmoothDeriv(BaseRegularization):
    """
    Smooth Regularization. This base class regularizes on the first
    spatial derivative, optionally normalized by the base cell size.

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
    """

    def __init__(self, mesh, orientation="x", **kwargs):
        self._length_scales = None
        self._normalized_gradients: bool = True
        self.orientation = orientation

        assert self.orientation in [
            "x",
            "y",
            "z",
        ], "Orientation must be 'x', 'y' or 'z'"

        if self.orientation == "y":
            assert mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif self.orientation == "z":
            assert mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )

        super().__init__(mesh=mesh, **kwargs)

    _reference_model_in_smooth: bool = False

    @property
    def reference_model_in_smooth(self) -> bool:
        """
        Use the reference model in the model gradient penalties.
        """
        return self._reference_model_in_smooth

    @reference_model_in_smooth.setter
    def reference_model_in_smooth(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                "'reference_model_in_smooth must be of type 'bool'. "
                f"{value} of type {type(value)} provided."
            )
        self._reference_model_in_smooth = value

    @property
    def _multiplier_pair(self):
        return "alpha_{orientation}".format(orientation=self.orientation)

    @property
    def W(self):
        """
        Weighting matrix that takes the first spatial difference
        with length scales in the specified orientation.
        """
        Ave = getattr(self.regularization_mesh, "aveCC2F{}".format(self.orientation))
        W = utils.sdiag(self.length_scales) * getattr(
            self.regularization_mesh,
            "cellDiff{orientation}Stencil".format(orientation=self.orientation),
        )
        if self.cell_weights is not None:

            W = utils.sdiag((Ave * (self.cell_weights)) ** 0.5) * W
        else:
            W = utils.sdiag((Ave * self.regularization_mesh.vol) ** 0.5) * W

        return W

    @property
    def length_scales(self):
        """
        Normalized cell based weighting

        """
        if getattr(self, "_length_scales", None) is None:
            Ave = getattr(self.regularization_mesh, "aveCC2F{}".format(self.orientation))
            index = "xyz".index(self.orientation)
            length_scales = Ave * (
                    self.regularization_mesh.Pac.T * self.regularization_mesh.mesh.h_gridded[:, index]
            )

            if self.normalized_gradients:
                length_scales /= length_scales.min()

            self._length_scales = length_scales**-1.0

        return self._length_scales

    @length_scales.setter
    def length_scales(self, value):
        self._length_scales = value

    @property
    def normalized_gradients(self):
        """
        Pre-normalize the model gradients by the base cell size
        """
        return self._normalized_gradients

    @normalized_gradients.setter
    def normalized_gradients(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "'normalized_gradients must be of type 'bool'. "
                f"{value} of type {type(value)} provided."
            )
        self._normalized_gradients = value

###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################
class ComboRegularization(ComboObjectiveFunction):
    def __init__(self, mesh, objfcts=[], **kwargs):

        super().__init__(
            objfcts=objfcts, multipliers=None
        )
        self.regularization_mesh = RegularizationMesh(mesh)
        if "active_cells" in kwargs.keys():
            active_cells = kwargs.pop("active_cells")
            self.regularization_mesh.active_cells = active_cells
        utils.setKwargs(self, **kwargs)

        # link these attributes
        linkattrs = [
            "regularization_mesh",
            "active_cells",
            "cell_weights",
            "mapping"
        ]

        for attr in linkattrs:
            val = getattr(self, attr)
            if val is not None:
                [setattr(fct, attr, val) for fct in self.objfcts]

    # Properties
    alpha_s = props.Float("smallness weight")
    alpha_x = props.Float("weight for the first x-derivative")
    alpha_y = props.Float("weight for the first y-derivative")
    alpha_z = props.Float("weight for the first z-derivative")
    alpha_xx = props.Float("weight for the second x-derivative")
    alpha_yy = props.Float("weight for the second y-derivative")
    alpha_zz = props.Float("weight for the second z-derivative")

    counter = None

    reference_model = props.Array("reference model")
    reference_model_in_smooth = properties.Bool(
        "include reference_model in the smoothness calculation?", default=False
    )
    active_cells = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    cell_weights = properties.Array(
        "regularization weights applied at cell centers", dtype=float
    )
    scale = properties.Float("function scaling applied inside the norm", default=1.0)
    regularization_mesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        maps.IdentityMap,
        default=maps.IdentityMap(),
    )

    @property
    def cell_weights(self):
        """Regularization weights applied at cell centers"""
        return self._cell_weights

    @cell_weights.setter
    def cell_weights(self, value):
        if not isinstance(value, np.ndarray) and not value.dtype == float:
            TypeError(
                "'cell_weights' must by a {} of type float. "
                f"A {type(value)} was provided."
            )
        for fct in self.objfcts:
            fct.cell_weights = value

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self.mapping, "nP") != "*":
            return self.mapping.nP
        elif getattr(self.regularization_mesh, "nC") != "*":
            return self.regularization_mesh.nC
        else:
            return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        nC = getattr(self.regularization_mesh, "nC", None)
        mapping = getattr(self, "mapping", None)

        if nC != "*" and nC is not None:
            return self.regularization_mesh.nC
        elif mapping is not None and mapping.shape[0] != "*":
            return self.mapping.shape[0]
        else:
            return self.nP

    def _delta_m(self, m):
        if self.reference_model is None:
            return m
        return -self.reference_model + m  # in case self.reference_model is Zero, returns type m

    @property
    def multipliers(self):
        """
        Factors that multiply the objective functions that are summed together
        to build to composite regularization
        """
        return [
            getattr(self, "{alpha}".format(alpha=objfct._multiplier_pair))
            for objfct in self.objfcts
        ]

    # Observers and Validators
    @properties.validator("active_cells")
    def _cast_to_bool(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":  # cast it to a bool otherwise
                tmp = value
                value = np.zeros(self.regularization_mesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

        # update regularization_mesh active_cells
        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = utils.mkvc(value)

    @properties.observer("active_cells")
    def _update_regularization_mesh_active_cells(self, change):
        # update regularization_mesh active_cells
        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = change["value"]

    @properties.observer("reference_model")
    def _mirror_reference_model_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, "reference_model_in_smooth", None) is not None:
                if self.reference_model_in_smooth is False:
                    fct.reference_model = utils.Zero()
                else:
                    fct.reference_model = change["value"]
            else:
                fct.reference_model = change["value"]

    @properties.observer("reference_model_in_smooth")
    def _mirror_reference_model_in_smooth_to_objfctlist(self, change):
        for fct in self.objfcts:
            if getattr(fct, "reference_model_in_smooth", None) is not None:
                fct.reference_model_in_smooth = change["value"]

    @properties.observer("active_cells")
    def _mirror_active_cells_to_objfctlist(self, change):
        value = change["value"]
        if value is not None:
            if value.dtype != "bool":
                tmp = value
                value = np.zeros(self.mesh.nC, dtype=bool)
                value[tmp] = True
                change["value"] = value

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = value

        for fct in self.objfcts:
            fct.active_cells = value

    @mapping.setter
    def mapping(self, value):
        for fct in self.objfcts:
            fct.mapping = value

    @properties.validator("cell_weights")
    def _validate_cell_weights(self, change):
        if change["value"] is not None:
            # todo: residual size? we need to know the expected end shape
            if self._nC_residual != "*":
                assert (
                        len(change["value"]) == self._nC_residual
                ), "cell_weights must be length {} not {}".format(
                    self._nC_residual, len(change["value"])
                )



class Simple(ComboRegularization):

    """
    Simple regularization that does not include length scales in the
    derivatives.

    .. math::

        r(\mathbf{m}) = \\alpha_s \phi_s + \\alpha_x \phi_x +
        \\alpha_y \phi_y + \\alpha_z \phi_z

    where:

    - :math:`\phi_s` is a :class:`SimPEG.regularization.Small` instance
    - :math:`\phi_x` is a :class:`SimPEG.regularization.SmoothDeriv` instance, with :code:`orientation='x'`
    - :math:`\phi_y` is a :class:`SimPEG.regularization.SmoothDeriv` instance, with :code:`orientation='y'`
    - :math:`\phi_z` is a :class:`SimPEG.regularization.SmoothDeriv` instance, with :code:`orientation='z'`


    **Required Inputs**

    :param discretize.base.BaseMesh mesh: a SimPEG mesh

    **Optional Inputs**

    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)

    **Weighting Parameters**

    :param float alpha_s: weighting on the smallness (default 1.)
    :param float alpha_x: weighting on the x-smoothness (default 1.)
    :param float alpha_y: weighting on the y-smoothness (default 1.)
    :param float alpha_z: weighting on the z-smoothness(default 1.)

    """

    def __init__(
        self, mesh, alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs
    ):

        objfcts = [
            Small(mesh=mesh, **kwargs),
            SmoothDeriv(mesh=mesh, orientation="x", **kwargs),
        ]

        if mesh.dim > 1:
            objfcts.append(SmoothDeriv(mesh=mesh, orientation="y", **kwargs))

        if mesh.dim > 2:
            objfcts.append(SmoothDeriv(mesh=mesh, orientation="z", **kwargs))

        super(Simple, self).__init__(
            mesh=mesh,
            objfcts=objfcts,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            **kwargs
        )