from __future__ import annotations

import numpy as np
from discretize.base import BaseMesh
import warnings
from typing import TYPE_CHECKING
from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import RegularizationMesh

from SimPEG.utils.code_utils import deprecate_property, validate_ndarray_with_shape

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseRegularization(BaseObjectiveFunction):
    """
    Base class for regularization. Inherit this for building your own
    regularization. The base regularization assumes a weighted l2-norm style of
    regularization. However, if you wish to employ a different norm, the
    methods :meth:`__call__`, :meth:`deriv` and :meth:`deriv2` can be
    over-written

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param active_cells: Array of bool defining the set of active cells.
    :param mapping: Model map
    :param reference_model: Array of model values used to constrain the inversion
    :param units: Model units identifier. Special case for 'radian'
    :param weights: Weight multipliers to customize the least-squares function.
    """

    _model = None

    def __init__(
        self,
        mesh: RegularizationMesh | BaseMesh,
        active_cells: np.ndarray | None = None,
        mapping: maps.IdentityMap | None = None,
        reference_model: np.ndarray | None = None,
        units: str | None = None,
        weights: dict | None = None,
        **kwargs,
    ):
        if isinstance(mesh, BaseMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            raise TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )

        self._regularization_mesh = mesh
        self._weights = {}

        if active_cells is not None:
            self.active_cells = active_cells

        self.mapping = mapping

        super().__init__(**kwargs)

        self.reference_model = reference_model
        self.units = units

        if weights is not None:
            if not isinstance(weights, dict):
                weights = {"user_weights": weights}
            self.set_weights(**weights)

    @property
    def active_cells(self) -> np.ndarray:
        """A boolean array of active cells on the regularization

        Returns
        -------
        (n_cells, ) Array of bool

        Notes
        -----
        If this is set with an array of integers, it interprets it as an array
        listing the active cell indices.
        """
        return self.regularization_mesh.active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray | None):
        self.regularization_mesh.active_cells = values

        if values is not None:
            volume_term = "volume" in self._weights
            self._weights = {}
            self._W = None
            if volume_term:
                self.set_weights(volume=self.regularization_mesh.vol)

    indActive = deprecate_property(
        active_cells,
        "indActive",
        "0.x.0",
        error=False,
    )

    @property
    def model(self) -> np.ndarray:
        """Physical property model"""
        return self._model

    @model.setter
    def model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        values = validate_ndarray_with_shape(
            "model", values, shape=(self._nC_residual,), dtype=float
        )

        self._model = values

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping applied to the model values"""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: maps.IdentityMap):
        if mapping is None:
            mapping = maps.IdentityMap()
        if not isinstance(mapping, maps.IdentityMap):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )
        self._mapping = mapping

    @property
    def units(self) -> str | None:
        """Specify the model units. Special care given to 'radian' values"""
        return self._units

    @units.setter
    def units(self, units: str | None):
        if units is not None and not isinstance(units, str):
            raise TypeError(
                f"'units' must be None or type str. "
                f"Value of type {type(units)} provided."
            )
        self._units = units

    @property
    def _weights_shapes(self) -> tuple[int] | str:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        if (
            getattr(self, "_regularization_mesh", None) is not None
            and self.regularization_mesh.nC != "*"
        ):
            return (self.regularization_mesh.nC,)

        if getattr(self, "_mapping", None) is not None and self.mapping.shape != "*":
            return (self.mapping.shape[0],)

        return ("*",)

    @property
    def reference_model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):
        if values is not None:
            if isinstance(values, float):
                values = np.ones(self._nC_residual) * values

            values = validate_ndarray_with_shape(
                "reference_model", values, shape=(self._nC_residual,), dtype=float
            )
        self._reference_model = values

    mref = deprecate_property(
        reference_model,
        "mref",
        "0.x.0",
        error=False,
    )

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh"""
        return self._regularization_mesh

    regmesh = deprecate_property(
        regularization_mesh,
        "regmesh",
        "0.x.0",
        error=False,
    )

    @property
    def cell_weights(self) -> np.ndarray:
        """Deprecated property for 'volume' and user defined weights."""
        warnings.warn(
            "cell_weights are deprecated please access weights using the `set_weights`,"
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.x.0",
            DeprecationWarning,
        )
        return np.prod(list(self._weights.values()), axis=0)

    @cell_weights.setter
    def cell_weights(self, value):
        warnings.warn(
            "cell_weights are deprecated please access weights using the `set_weights`,"
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.x.0",
            DeprecationWarning,
        )
        self.set_weights(cell_weights=value)

    def get_weights(self, key) -> np.ndarray:
        """Weights for a given key."""
        return self._weights[key]

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights to the regularization

        Parameters:
        -----------
        **kwargs : key, numpy.ndarray
            Each keyword argument is added to the weights used by the regularization.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        """
        for key, values in weights.items():
            values = validate_ndarray_with_shape(
                "weights", values, shape=self._weights_shapes, dtype=float
            )
            self._weights[key] = values
        self._W = None

    def remove_weights(self, key):
        """Removes the weights with a given key"""
        try:
            self._weights.pop(key)
        except KeyError as error:
            raise KeyError(f"{key} is not in the weights dictionary") from error
        self._W = None

    @property
    def W(self) -> np.ndarray:
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = utils.sdiag(weights ** 0.5)
        return self._W

    @property
    def _nC_residual(self) -> int:
        """
        Shape of the residual
        """

        nC = getattr(self.regularization_mesh, "nC", None)
        mapping = getattr(self, "_mapping", None)

        if mapping is not None and mapping.shape[1] != "*":
            return self.mapping.shape[1]

        if nC != "*" and nC is not None:
            return self.regularization_mesh.nC

        return self._weights_shapes[0]

    def _delta_m(self, m) -> np.ndarray:
        if self.reference_model is None:
            return m
        return (
            m - self.reference_model
        )  # in case self.reference_model is Zero, returns type m

    @utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2} \\| \\mathbf{W} \\mathbf{f(m)} \\|_2^2
        """
        r = self.W * self.f_m(m)
        return 0.5 * r.dot(r)

    def f_m(self, m) -> np.ndarray:
        raise AttributeError("Regularization class must have a 'f_m' implementation.")

    def f_m_deriv(self, m) -> csr_matrix:
        raise AttributeError(
            "Regularization class must have a 'f_m_deriv' implementation."
        )

    @utils.timeIt
    def deriv(self, m) -> np.ndarray:
        """

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \\mathbf{W^\\top W (m-m_\\text{ref})}

        """
        r = self.W * self.f_m(m)
        return self.f_m_deriv(m).T * (self.W.T * r)

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        """
        Second derivative

        :param numpy.ndarray m: geophysical model
        :param numpy.ndarray v: vector to multiply
        :rtype: scipy.sparse.csr_matrix
        :return: WtW, or if v is supplied WtW*v (numpy.ndarray)

        The regularization is:

        .. math::

            R(m) = \\frac{1}{2}\\mathbf{(m-m_\\text{ref})^\\top W^\\top
            W(m-m_\\text{ref})}

        So the second derivative is straight forward:

        .. math::

            R(m) = \\mathbf{W^\\top W}

        """
        f_m_deriv = self.f_m_deriv(m)
        if v is None:
            return f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)

        return f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))


class Smallness(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model.

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{V}^T
            \\mathbf{W}^T
        \\mathbf{W} \\mathbf{V} (\\mathbf{m} - \\mathbf{m_{ref}})

    where
    :math:`\\mathbf{m}` is the model,
    :math:`\\mathbf{m_{ref}}` is a reference model,
    :math:`\\mathbf{V}` are square root of cell volumes and
    :math:`\\mathbf{W}` is a weighting matrix (default Identity). If fixed or
        free weights are provided, then it is :code:`diag(np.sqrt(weights))`).


    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int shape: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from
        model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size
        of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)

    def f_m(self, m) -> np.ndarray:
        """
        Model residual
        """
        return self.mapping * self._delta_m(m)

    def f_m_deriv(self, m) -> csr_matrix:
        """
        Derivative of the model residual
        """
        return self.mapping.deriv(self._delta_m(m))


class SmoothnessFirstOrder(BaseRegularization):
    """
    Smooth Regularization. This base class regularizes on the first
    spatial derivative, optionally normalized by the base cell size.

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param IdentityMap mapping: regularization mapping, takes the model from
        model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the
        size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the
        smoothness computation? (eg. look at Deriv of m (False) or Deriv of
        (m-reference_model) (True))
    :param numpy.ndarray weights: vector of cell weights (applied in all terms)
    """

    def __init__(
        self, mesh, orientation="x", reference_model_in_smooth=False, **kwargs
    ):
        self.reference_model_in_smooth = reference_model_in_smooth

        if orientation not in ["x", "y", "z"]:
            raise ValueError("Orientation must be 'x', 'y' or 'z'")

        if orientation == "y" and mesh.dim < 2:
            raise ValueError(
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction."
            )
        elif orientation == "z" and mesh.dim < 3:
            raise ValueError(
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )
        self._orientation = orientation

        super().__init__(mesh=mesh, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)

    @property
    def _weights_shapes(self):
        """Acceptable lengths for the weights

        Returns
        -------
        tuple
            A tuple of each acceptable lengths for the weights
        """
        n_active_f, n_active_c = getattr(
            self.regularization_mesh, "aveCC2F{}".format(self.orientation)
        ).shape
        return [(n_active_f,), (n_active_c,)]

    @property
    def cell_gradient(self):
        """Cell gradient operator"""
        if getattr(self, "_cell_gradient", None) is None:
            self._cell_gradient = getattr(
                self.regularization_mesh, "cell_gradient_{}".format(self.orientation)
            )
        return self._cell_gradient

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
                f"Value of type {type(value)} provided."
            )
        self._reference_model_in_smooth = value

    def _delta_m(self, m):
        if self.reference_model is None or not self.reference_model_in_smooth:
            return m
        return m - self.reference_model

    @property
    def _multiplier_pair(self):
        return f"alpha_{self.orientation}"

    def f_m(self, m):
        """
        Model gradient
        """
        dfm_dl = self.cell_gradient @ (self.mapping * self._delta_m(m))

        if self.units is not None and self.units.lower() == "radian":
            return (
                utils.mat_utils.coterminal(dfm_dl * self._cell_distances)
                / self._cell_distances
            )
        return dfm_dl

    def f_m_deriv(self, m) -> csr_matrix:
        """
        Derivative of the model gradient
        """
        return self.cell_gradient @ self.mapping.deriv(self._delta_m(m))

    @property
    def W(self):
        """
        Weighting matrix that takes the volumes, free weights, fixed weights and
        length scales of the difference operator (normalized optional).
        """
        if getattr(self, "_W", None) is None:
            average_cell_2_face = getattr(
                self.regularization_mesh, "aveCC2F{}".format(self.orientation)
            )
            weights = 1.0
            for values in self._weights.values():
                if values.shape[0] == self.regularization_mesh.nC:
                    values = average_cell_2_face * values
                weights *= values
            self._W = utils.sdiag(weights ** 0.5)
        return self._W

    @property
    def _cell_distances(self):
        """
        Distances between cell centers for the cell center difference.
        """
        return getattr(self.regularization_mesh, f"cell_distances_{self.orientation}")

    @property
    def orientation(self):
        return self._orientation


class SmoothnessSecondOrder(SmoothnessFirstOrder):
    """
    This base class regularizes on the second
    spatial derivative, optionally normalized by the base cell size.

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from
        model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the
        size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the
        smoothness computation? (eg. look at Deriv of m (False) or Deriv of
        (m-reference_model) (True))
    :param numpy.ndarray weights: vector of cell weights (applied in all terms)
    """

    def f_m(self, m):
        """
        Second model derivative
        """
        dfm_dl = self.cell_gradient @ (self.mapping * self._delta_m(m))

        if self.units is not None and self.units.lower() == "radian":
            dfm_dl = (
                utils.mat_utils.coterminal(dfm_dl * self.length_scales)
                / self.length_scales
            )

        dfm_dl2 = self.cell_gradient.T @ dfm_dl

        return dfm_dl2

    def f_m_deriv(self, m) -> csr_matrix:
        """
        Derivative of the second model residual
        """
        return (
            self.cell_gradient.T
            @ self.cell_gradient
            @ self.mapping.deriv(self._delta_m(m))
        )

    @property
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = utils.sdiag(weights ** 0.5)

        return self._W

    @property
    def _multiplier_pair(self):
        return f"alpha_{self.orientation}{self.orientation}"


###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################


class WeightedLeastSquares(ComboObjectiveFunction):
    _model = None

    def __init__(
        self,
        mesh,
        active_cells=None,
        alpha_s=1.0,
        alpha_x=None,
        alpha_y=None,
        alpha_z=None,
        alpha_xx=0.0,
        alpha_yy=0.0,
        alpha_zz=0.0,
        length_scale_x=None,
        length_scale_y=None,
        length_scale_z=None,
        mapping=None,
        reference_model=None,
        reference_model_in_smooth=False,
        weights=None,
        **kwargs,
    ):

        if isinstance(mesh, BaseMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh
        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        self.alpha_s = alpha_s
        if alpha_x is not None:
            if length_scale_x is not None:
                raise ValueError(
                    "Attempted to set both alpha_x and length_scale_x at the same time. Please "
                    "use only one of them"
                )
            self.alpha_x = alpha_x
        else:
            self.length_scale_x = length_scale_x

        if alpha_y is not None:
            if length_scale_y is not None:
                raise ValueError(
                    "Attempted to set both alpha_y and length_scale_y at the same time. Please "
                    "use only one of them"
                )
            self.alpha_y = alpha_y
        else:
            self.length_scale_y = length_scale_y

        if alpha_z is not None:
            if length_scale_z is not None:
                raise ValueError(
                    "Attempted to set both alpha_z and length_scale_z at the same time. Please "
                    "use only one of them"
                )
            self.alpha_z = alpha_z
        else:
            self.length_scale_z = length_scale_z

        # do this to allow child classes to also pass a list of objfcts to this constructor
        if "objfcts" not in kwargs:
            objfcts = [
                Smallness(mesh=self.regularization_mesh),
                SmoothnessFirstOrder(mesh=self.regularization_mesh, orientation="x"),
                SmoothnessSecondOrder(mesh=self.regularization_mesh, orientation="x"),
            ]

            if mesh.dim > 1:
                objfcts.extend(
                    [
                        SmoothnessFirstOrder(
                            mesh=self.regularization_mesh, orientation="y"
                        ),
                        SmoothnessSecondOrder(
                            mesh=self.regularization_mesh, orientation="y"
                        ),
                    ]
                )

            if mesh.dim > 2:
                objfcts.extend(
                    [
                        SmoothnessFirstOrder(
                            mesh=self.regularization_mesh, orientation="z"
                        ),
                        SmoothnessSecondOrder(
                            mesh=self.regularization_mesh, orientation="z"
                        ),
                    ]
                )
        else:
            objfcts = kwargs.pop("objfcts")
        super().__init__(objfcts=objfcts, **kwargs)
        self.mapping = mapping
        self.reference_model = reference_model
        self.reference_model_in_smooth = reference_model_in_smooth
        self.alpha_xx = alpha_xx
        self.alpha_yy = alpha_yy
        self.alpha_zz = alpha_zz
        if weights is not None:
            if not isinstance(weights, dict):
                weights = {"user_weights": weights}
            self.set_weights(**weights)

    def set_weights(self, **weights):
        """Update weights in children objective functions"""
        for fct in self.objfcts:
            fct.set_weights(**weights)

    def remove_weights(self, key):
        """removes weights in children objective functions"""
        for fct in self.objfcts:
            fct.remove_weights(key)

    @property
    def cell_weights(self):
        # All of the objective functions should have the same weights,
        # so just grab the one from smallness here, which should also
        # trigger the deprecation warning
        return self.objfcts[0].cell_weights

    @cell_weights.setter
    def cell_weights(self, value):
        warnings.warn(
            "cell_weights are deprecated please access weights using the `set_weights`,"
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.x.0",
            DeprecationWarning,
        )
        self.set_weights(cell_weights=value)

    @property
    def alpha_s(self):
        """smallness weight"""
        return self._alpha_s

    @alpha_s.setter
    def alpha_s(self, value):
        if value is None:
            value = 1.0
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_s must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_s must be non-negative, not {value}")
        self._alpha_s = value

    @property
    def alpha_x(self):
        """weight for the first x-derivative"""
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_x must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_x must be non-negative, not {value}")
        self._alpha_x = value

    @property
    def alpha_y(self):
        """weight for the first y-derivative"""
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_y must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_y must be non-negative, not {value}")
        self._alpha_y = value

    @property
    def alpha_z(self):
        """weight for the first z-derivative"""
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, value):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_z must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_z must be non-negative, not {value}")
        self._alpha_z = value

    @property
    def alpha_xx(self):
        """weight for the second x-derivative"""
        return self._alpha_xx

    @alpha_xx.setter
    def alpha_xx(self, value):
        if value is None:
            value = (self.length_scale_x * self.regularization_mesh.base_length) ** 4.0
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_xx must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_xx must be non-negative, not {value}")
        self._alpha_xx = value

    @property
    def alpha_yy(self):
        """weight for the second y-derivative"""
        return self._alpha_yy

    @alpha_yy.setter
    def alpha_yy(self, value):
        if value is None:
            value = (self.length_scale_y * self.regularization_mesh.base_length) ** 4.0
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_yy must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_yy must be non-negative, not {value}")
        self._alpha_yy = value

    @property
    def alpha_zz(self):
        """weight for the second z-derivative"""
        return self._alpha_zz

    @alpha_zz.setter
    def alpha_zz(self, value):
        if value is None:
            value = (self.length_scale_z * self.regularization_mesh.base_length) ** 4.0
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"alpha_zz must be a real number, saw type{type(value)}")
        if value < 0:
            raise ValueError(f"alpha_zz must be non-negative, not {value}")
        self._alpha_zz = value

    @property
    def length_scale_x(self):
        """Constant multiplier of the base length scale on model gradients along x."""
        return np.sqrt(self.alpha_x / self.regularization_mesh.base_length)

    @length_scale_x.setter
    def length_scale_x(self, value: float):
        if value is None:
            value = 1.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"length_scale_x must be a real number, saw type{type(value)}"
            )
        self.alpha_x = (value * self.regularization_mesh.base_length) ** 2

    @property
    def length_scale_y(self):
        """Constant multiplier of the base length scale on model gradients along y."""
        return np.sqrt(self.alpha_y / self.regularization_mesh.base_length)

    @length_scale_y.setter
    def length_scale_y(self, value: float):
        if value is None:
            value = 1.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"length_scale_y must be a real number, saw type{type(value)}"
            )
        self.alpha_y = (value * self.regularization_mesh.base_length) ** 2

    @property
    def length_scale_z(self):
        """Constant multiplier of the base length scale on model gradients along z."""
        return np.sqrt(self.alpha_z / self.regularization_mesh.base_length)

    @length_scale_z.setter
    def length_scale_z(self, value: float):
        if value is None:
            value = 1.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"length_scale_z must be a real number, saw type{type(value)}"
            )
        self.alpha_z = (value * self.regularization_mesh.base_length) ** 2

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
                f"Value of type {type(value)} provided."
            )
        self._reference_model_in_smooth = value
        for fct in self.objfcts:
            if getattr(fct, "reference_model_in_smooth", None) is not None:
                fct.reference_model_in_smooth = value

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self, "mapping", None) is not None and self.mapping.nP != "*":
            return self.mapping.nP
        elif (
            getattr(self, "_regularization_mesh", None) is not None
            and self.regularization_mesh.nC != "*"
        ):
            return self.regularization_mesh.nC
        else:
            return "*"

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """

        nC = getattr(self.regularization_mesh, "nC", None)
        mapping = getattr(self, "_mapping", None)

        if mapping is not None and mapping.shape[1] != "*":
            return self.mapping.shape[1]
        elif nC != "*" and nC is not None:
            return self.regularization_mesh.nC
        else:
            return self.nP

    def _delta_m(self, m):
        if self.reference_model is None:
            return m
        return m - self.reference_model

    @property
    def multipliers(self):
        """
        Factors that multiply the objective functions that are summed together
        to build to composite regularization
        """
        return [getattr(self, objfct._multiplier_pair) for objfct in self.objfcts]

    @property
    def active_cells(self) -> np.ndarray:
        """Indices of active cells in the mesh"""
        return self.regularization_mesh.active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        self.regularization_mesh.active_cells = values
        active_cells = self.regularization_mesh.active_cells
        # notify the objtecive functions that the active_cells changed
        for objfct in self.objfcts:
            objfct.active_cells = active_cells

    indActive = deprecate_property(
        active_cells,
        "indActive",
        "0.x.0",
        error=False,
    )

    @property
    def reference_model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        for fct in self.objfcts:
            fct.reference_model = values

        self._reference_model = values

    mref = deprecate_property(
        reference_model,
        "mref",
        "0.x.0",
        error=False,
    )

    @property
    def model(self) -> np.ndarray:
        """Physical property model"""
        return self._model

    @model.setter
    def model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        for fct in self.objfcts:
            fct.model = values

        self._model = values

    @property
    def units(self) -> str:
        """Specify the model units. Special care given to 'radian' values"""
        return self._units

    @units.setter
    def units(self, units: str | None):
        if units is not None and not isinstance(units, str):
            raise TypeError(
                f"'units' must be None or type str. "
                f"Value of type {type(units)} provided."
            )
        for fct in self.objfcts:
            fct.units = units
        self._units = units

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh"""
        return self._regularization_mesh

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping applied to the model values"""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: maps.IdentityMap):
        if mapping is None:
            mapping = maps.IdentityMap(nP=self._nC_residual)

        if not isinstance(mapping, maps.IdentityMap):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )
        self._mapping = mapping

        for fct in self.objfcts:
            fct.mapping = mapping


###############################################################################
#                                                                             #
#                        Base Coupling Regularization                         #
#                                                                             #
###############################################################################
class BaseSimilarityMeasure(BaseRegularization):

    """
    Base class for the similarity term in joint inversions. Inherit this for building
    your own similarity term.  The BaseSimilarityMeasure assumes two different
    geophysical models through one similarity term. However, if you wish
    to combine more than two models, e.g., 3 models,
    you may want to add a total of three coupling terms:

    e.g., lambda1*(m1, m2) + lambda2*(m1, m3) + lambda3*(m2, m3)

    where, lambdas are weights for coupling terms. m1, m2 and m3 indicate
    three different models.
    """

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, **kwargs)
        self.wire_map = wire_map

    @property
    def wire_map(self):
        return self._wire_map

    @wire_map.setter
    def wire_map(self, wires):

        try:
            m1, m2 = wires.maps  # Assume a map has been passed for each model.
        except ValueError:
            ValueError("Wire map must have two model mappings")

        if m1[1].shape[0] != m2[1].shape[0]:
            raise ValueError(
                f"All models must be the same size! Got {m1[1].shape[0]} and {m2[1].shape[0]}"
            )
        self._wire_map = wires

    @property
    def nP(self):
        """
        number of model parameters
        """
        return self.wire_map.nP

    def deriv(self, model):
        """
        First derivative of the coupling term with respect to individual models.
        Returns an array of dimensions [k*M,1],
        k: number of models we are inverting for.
        M: number of cells in each model.

        """
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self, model, v=None):
        """
        Second derivative of the coupling term with respect to individual models.
        Returns either an array of dimensions [k*M,1] (v is not None), or
        sparse matrix of dimensions [k*M, k*M] (v is None).
        k: number of models we are inverting for.
        M: number of cells in each model.

        """
        raise NotImplementedError(
            "The method _deriv2 has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    @property
    def _nC_residual(self):
        """
        Shape of the residual
        """
        return self.wire_map.nP

    def __call__(self, model):
        """Returns the computed value of the coupling term."""
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )
