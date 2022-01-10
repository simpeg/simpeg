from __future__ import annotations

import numpy as np

from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import RegularizationMesh
from SimPEG.utils.code_utils import deprecate_property


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
    _mapping = None
    _model = None
    _reference_model = None
    _regularization_mesh = None
    _shape = None
    _units = None
    _weights: {} = None
    _W = None

    def __init__(self, mesh, active_cells=None, mapping=None, reference_model=None, units=None, **kwargs):
        self.regularization_mesh = RegularizationMesh(mesh)
        self.active_cells = active_cells
        self.regularization_mesh.active_cells = self.active_cells
        self.mapping = mapping
        self.reference_model = reference_model
        self.units = units

        super().__init__(**kwargs)

    # Properties
    @property
    def active_cells(self) -> np.ndarray:
        """Indices of active cells in the mesh"""
        return self._active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        validate_array_type("active_cells", values, bool)
        validate_shape("active_cells", values, self.regularization_mesh.nC)

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = values

        self._active_cells = values

    indActive = deprecate_property(
        active_cells,
        "indActive",
        new_name="active_cells",
        removal_version="0.x.0",
        future_warn=True,
    )

    @property
    def weights(self):
        """Regularization weights applied to the target elements"""
        raise AttributeError("Regularization class must have 'weights' implementation.")

    @property
    def model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._model

    @model.setter
    def model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        validate_array_type("model", values, float)
        validate_shape("model", values, self._nC_residual)
        self._model = values

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping applied to the model values"""
        if getattr(self, "_mapping", None) is None:
            self.mapping = maps.IdentityMap(nP=self._nC_residual)
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: maps.IdentityMap):
        if not isinstance(mapping, (maps.IdentityMap, type(None))):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )

        self._mapping = mapping

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
        self._units = units

    @property
    def shape(self):
        """
        number of model parameters
        """
        if getattr(self, "_regularization_mesh", None) is not None and self.regularization_mesh.nC != "*":
            return self.regularization_mesh.nC,
        elif getattr(self, "_mapping", None) is not None and self.mapping.shape != "*":
            return self.mapping.shape
        else:
            return "*"

    @property
    def reference_model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        validate_array_type("reference_model", values, float)
        validate_shape("reference_model", values, self._nC_residual)
        self._reference_model = values

    mref = deprecate_property(
        reference_model,
        "mref",
        new_name="reference_model",
        removal_version="0.x.0",
        future_warn=True,
    )

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh"""
        return self._regularization_mesh

    @regularization_mesh.setter
    def regularization_mesh(self, mesh: RegularizationMesh):
        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh

    regmesh = deprecate_property(
        regularization_mesh,
        "regmesh",
        new_name="regularization_mesh",
        removal_version="0.x.0",
        future_warn=True,
    )

    @property
    def W(self):
        """
        Weighting matrix
        """
        raise AttributeError("Regularization class must have a 'W' implementation.")

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
            return self.shape[0]

    def _delta_m(self, m):
        if self.reference_model is None:
            return m
        return m - self.reference_model  # in case self.reference_model is Zero, returns type m

    @utils.timeIt
    def __call__(self, m):
        """
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \\frac{1}{2}
        """
        r = self.W * self.f_m(m)
        return 0.5 * r.dot(r)

    def f_m(self, m):
        raise AttributeError("Regularization class must have a 'f_m' implementation.")

    def f_m_deriv(self, m):
        raise AttributeError("Regularization class must have a 'f_m_deriv' implementation.")

    @utils.timeIt
    def deriv(self, m):
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
        f_m_deriv = self.f_m_deriv(m)
        if v is None:
            return f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)

        return f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))


class Small(BaseRegularization):
    """
    Small regularization - L2 regularization on the difference between a
    model and a reference model.

    .. math::

        r(m) = \\frac{1}{2}(\\mathbf{m} - \\mathbf{m_ref})^\top \\mathbf{V}^T \\mathbf{W}^T
        \\mathbf{W} \\mathbf{V} (\\mathbf{m} - \\mathbf{m_{ref}})

    where
    :math:`\\mathbf{m}` is the model,
    :math:`\\mathbf{m_{ref}}` is a reference model,
    :math:`\\mathbf{V}` are square root of cell volumes and
    :math:`\\mathbf{W}` is a weighting matrix (default Identity).
    If fixed or free weights are provided, then it is :code:`diag(np.sqrt(weights))`).


    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int shape: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, weights=None, **kwargs):

        super().__init__(**kwargs)

        self.weights = weights

    def f_m(self, m):
        """
        Model residual
        """
        return self.mapping * self._delta_m(m)

    def f_m_deriv(self, m):
        """
        Derivative of the model residual
        """
        return self.mapping.deriv(self._delta_m(m))

    @property
    def weights(self):
        """Regularization weights applied to the target elements"""
        if getattr(self, "_weights", None) is None:
            self._weights = {}
            self.add_set_weights(
                {"volume": self.regularization_mesh.vol}
            )

        return self._weights

    @weights.setter
    def weights(self, weights: dict[str, np.ndarray] | np.ndarray | None):
        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights["user_weights"] = weights

            if not isinstance(weights, dict):
                raise TypeError("Weights must be provided as a dictionary or None.")

            for key, values in weights.items():
                validate_array_type("weights", values, float)
                validate_shape("weights", values, self.shape[0])

        self._weights = weights
        self._W = None

    def add_set_weights(self, weights: dict):
        for key, values in weights.items():
            validate_array_type("weights", values, float)
            validate_shape("weights", values, self.shape[0])
            self.weights[key] = values
        self._W = None

    @property
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self.weights.values()), axis=0)
            self._W = utils.sdiag(weights ** 0.5)

        return self._W


class SmoothDeriv(BaseRegularization):
    """
    Smooth Regularization. This base class regularizes on the first
    spatial derivative, optionally normalized by the base cell size.

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int shape: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray weights: vector of cell weights (applied in all terms)
    """
    _cell_difference = None
    _length_scales = None
    _orientation = None
    _shape = None
    _reference_model_in_smooth: bool = False

    def __init__(self, mesh, orientation="x", reference_model_in_smooth=False, weights=None, **kwargs):
        self.reference_model_in_smooth = reference_model_in_smooth

        super().__init__(mesh=mesh, **kwargs)

        self.orientation = orientation
        self.weights = weights

    @property
    def shape(self):
        if getattr(self, "_nP", None) is None:
            self._shape = getattr(
                self.regularization_mesh, "aveCC2F{}".format(self.orientation)
            ).shape
        return self._shape

    @property
    def cell_difference(self):
        """Cell difference operator"""
        if getattr(self, "_cell_difference", None) is None:
            self._cell_difference = getattr(
                self.regularization_mesh, "cellDiff{}Stencil".format(self.orientation)
            )
        return self._cell_difference

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

    @property
    def _multiplier_pair(self):
        return f"alpha_{self.orientation}"

    def f_m(self, m):
        """
        Model gradient
        """
        if self.reference_model_in_smooth:
            delta_m = self._delta_m(m)
        else:
            delta_m = m

        dfm_dl = self.cell_difference @ (self.mapping * delta_m)

        if self.units == "radian":
            return utils.mat_utils.coterminal(dfm_dl)
        return dfm_dl

    def f_m_deriv(self, m):
        """
        Derivative of the model gradient
        """
        return self.cell_difference @ self.mapping.deriv(self._delta_m(m))

    @property
    def weights(self):
        """Regularization weights applied to the target elements"""
        if getattr(self, "_weights", None) is None:
            self._weights = {}
            self.add_set_weights(
                {
                    "volume": self.regularization_mesh.vol,
                    "length_scales": self.length_scales**-2.0,
                }
            )

        return self._weights

    @weights.setter
    def weights(self, weights: dict[str, np.ndarray] | np.ndarray | None):
        if weights is not None:

            if isinstance(weights, np.ndarray):
                weights["user_weights"] = weights

            if not isinstance(weights, dict):
                raise TypeError("Weights must be provided as a dictionary or None.")

            for key, values in weights.items():
                validate_array_type("weights", values, float)
                validate_shape("weights", values, self.shape[0])

        self._weights = weights
        self._W = None

    def add_set_weights(self, weights: dict):
        for key, values in weights.items():
            average_cell_2_face = getattr(
                self.regularization_mesh, "aveCC2F{}".format(self.orientation)
            )
            validate_array_type("weights", values, float)

            if values.shape[0] == self.regularization_mesh.nC:
                values = average_cell_2_face * values

            validate_shape("weights", values, self.shape[0])
            self.weights[key] = values

        self._W = None

    @property
    def W(self):
        """
        Weighting matrix that takes the volumes, free weights, fixed weights and
        length scales of the difference operator (normalized optional).
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self.weights.values()), axis=0)
            self._W = utils.sdiag(weights**0.5)

        return self._W

    @property
    def length_scales(self):
        """
        Length scales for the cell center difference.
        Normalized by the smallest cell dimension if 'normalized_gradient' is True.
        """
        if getattr(self, "_length_scales", None) is None:
            Ave = getattr(self.regularization_mesh, "aveCC2F{}".format(self.orientation))
            index = "xyz".index(self.orientation)
            self._length_scales = Ave * (
                    self.regularization_mesh.Pac.T * self.regularization_mesh.mesh.h_gridded[:, index]
            )

        return self._length_scales

    @length_scales.setter
    def length_scales(self, values):
        validate_array_type("length_scales", values, float)
        validate_shape("length_scales", values, self.nP)
        self._length_scales = values

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        assert value in [
            "x",
            "y",
            "z",
        ], "Orientation must be 'x', 'y' or 'z'"

        if value == "y":
            assert self.regularization_mesh.dim > 1, (
                "Mesh must have at least 2 dimensions to regularize along the "
                "y-direction"
            )

        elif value == "z":
            assert self.regularization_mesh.dim > 2, (
                "Mesh must have at least 3 dimensions to regularize along the "
                "z-direction"
            )
        self._orientation = value


class SmoothDeriv2(SmoothDeriv):
    """
    This base class regularizes on the second
    spatial derivative, optionally normalized by the base cell size.

    **Optional Inputs**

    :param discretize.base.BaseMesh mesh: SimPEG mesh
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray weights: vector of cell weights (applied in all terms)
    """
    def __init__(self, mesh, orientation="x", **kwargs):
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)

    def f_m(self, m):
        """
        Second model derivative
        """
        if self.reference_model_in_smooth:
            delta_m = self._delta_m(m)
        else:
            delta_m = m

        dfm_dl = self.cell_difference @ (self.mapping * delta_m)

        if self.units == "radian":
            dfm_dl = utils.mat_utils.coterminal(dfm_dl)

        dfm_dl2 = self.cell_difference.T @ (self.length_scales ** 2. * dfm_dl)

        return dfm_dl2

    def f_m_deriv(self, m):
        """
        Derivative of the second model residual
        """
        return (
            self.cell_difference.T *
            utils.sdiag(self.length_scales ** 2.) *
            self.cell_difference @ self.mapping.deriv(self._delta_m(m))
        )

    @property
    def W(self):
        """
        Weighting matrix to cell center.
        """
        weights = np.prod(list(self.weights.values()), axis=0)

        return utils.sdiag(weights ** 0.5)

    @property
    def _multiplier_pair(self):
        return f"alpha_{self.orientation}{self.orientation}"

###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################


class LeastSquaresRegularization(ComboObjectiveFunction):
    _active_cells = None
    _alpha_s = None
    _alpha_x = None
    _alpha_y = None
    _alpha_z = None
    _alpha_xx = None
    _alpha_yy = None
    _alpha_zz = None
    _model = None
    _mapping = None
    _normalized_gradients = True
    _reference_model_in_smooth = False
    _reference_model = None
    _regularization_mesh = None
    _units = None
    _weights = None

    def __init__(
        self,
        mesh,
        active_cells=None,
        alpha_s=None,
        alpha_x=None,
        alpha_y=None,
        alpha_z=None,
        alpha_xx=None,
        alpha_yy=None,
        alpha_zz=None,
        mapping=None,
        objfcts=None,
        reference_model=None,
        reference_model_in_smooth=False,
        **kwargs
    ):

        self.regularization_mesh = RegularizationMesh(mesh)

        if objfcts is None:
            objfcts = [
                Small(mesh=mesh),
                SmoothDeriv(mesh=mesh, orientation="x"),
            ]

            if mesh.dim > 1:
                objfcts.append(SmoothDeriv(mesh=mesh, orientation="y"))

            if mesh.dim > 2:
                objfcts.append(SmoothDeriv(mesh=mesh, orientation="z"))

        super().__init__(
            objfcts=objfcts,
            active_cells=active_cells,
            mapping=mapping,
            reference_model=reference_model,
            reference_model_in_smooth=reference_model_in_smooth,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            alpha_xx=alpha_xx,
            alpha_yy=alpha_yy,
            alpha_zz=alpha_zz,
            **kwargs
        )

    @property
    def alpha_s(self):
        """smallness weight"""
        if getattr(self, "_alpha_s", None) is None:
            self._alpha_s = 1.
        return self._alpha_s

    @alpha_s.setter
    def alpha_s(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_s' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_s = value

    @property
    def alpha_x(self):
        """weight for the first x-derivative"""
        if getattr(self, "_alpha_x", None) is None:
            self._alpha_x = self.regularization_mesh.mesh.h_gridded.min()**2.
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_x' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_x = value

    @property
    def alpha_y(self):
        """weight for the first y-derivative"""
        if getattr(self, "_alpha_y", None) is None:
            self._alpha_y = self.regularization_mesh.mesh.h_gridded.min()**2.
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_y' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_y = value

    @property
    def alpha_z(self):
        """weight for the first z-derivative"""
        if getattr(self, "_alpha_z", None) is None:
            self._alpha_z = self.regularization_mesh.mesh.h_gridded.min()**2.
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_z' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_z = value

    @property
    def alpha_xx(self):
        """weight for the second x-derivative"""
        if getattr(self, "_alpha_xx", None) is None:
            self._alpha_xx = self.regularization_mesh.mesh.h_gridded.min()**4.
        return self._alpha_xx

    @alpha_xx.setter
    def alpha_xx(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_xx' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_xx = value

    @property
    def alpha_yy(self):
        """weight for the second y-derivative"""
        if getattr(self, "_alpha_yy", None) is None:
            self._alpha_yy = self.regularization_mesh.mesh.h_gridded.min()**4.
        return self._alpha_yy

    @alpha_yy.setter
    def alpha_yy(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_yy' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_yy = value

    @property
    def alpha_zz(self):
        """weight for the second z-derivative"""
        if getattr(self, "_alpha_zz", None) is None:
            self._alpha_zz = self.regularization_mesh.mesh.h_gridded.min()**4.
        return self._alpha_zz

    @alpha_zz.setter
    def alpha_zz(self, value):
        if not isinstance(value, (float, type(None))) and value > 0:
            raise ValueError("Input 'alpha_zz' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_zz = value

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

    @property
    def weights(self):
        """Fixed regularization weights applied at cell centers"""
        return self._weights

    @weights.setter
    def weights(self, value: dict[str, np.ndarray] | np.ndarray | None):
        for fct in self.objfcts:
            fct.weights = value

    # Other properties and methods
    @property
    def nP(self):
        """
        number of model parameters
        """
        if getattr(self, "_mapping", None) is not None and self.mapping.nP != "*":
            return self.mapping.nP
        elif getattr(self, "_regularization_mesh", None) is not None and self.regularization_mesh.nC != "*":
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
        return m-self.reference_model

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
                f"Value of type {type(value)} provided."
            )
        self._normalized_gradients = value
        for fct in self.objfcts:
            if hasattr(fct, "_normalized_gradients"):
                fct.normalized_gradients = value

    @property
    def active_cells(self) -> np.ndarray:
        """Indices of active cells in the mesh"""
        return self._active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        validate_array_type("active_cells", values, bool)
        validate_shape("active_cells", values, self.nP)

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = utils.mkvc(values)

        for objfct in self.objfcts:
            objfct.active_cells = values

        self._active_cells = values

    indActive = deprecate_property(
        active_cells,
        "indActive",
        new_name="active_cells",
        removal_version="0.x.0",
        future_warn=True,
    )

    @property
    def reference_model(self) -> np.ndarray:
        """Reference physical property model"""
        return self._reference_model

    @reference_model.setter
    def reference_model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        validate_array_type("reference_model", values, float)
        validate_shape("reference_model", values, self._nC_residual)

        for fct in self.objfcts:
            fct.reference_model = values

        self._reference_model = values

    mref = deprecate_property(
        reference_model,
        "mref",
        new_name="reference_model",
        removal_version="0.x.0",
        future_warn=True,
    )

    @property
    def model(self) -> np.ndarray:
        """Physical property model"""
        return self._model

    @model.setter
    def model(self, values: np.ndarray | float):

        if isinstance(values, float):
            values = np.ones(self._nC_residual) * values

        validate_array_type("model", values, float)
        validate_shape("model", values, self._nC_residual)

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
            fct.model = units

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh"""
        return self._regularization_mesh

    @regularization_mesh.setter
    def regularization_mesh(self, mesh: RegularizationMesh):
        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping applied to the model values"""
        if getattr(self, "_mapping", None) is None:
            self._mapping = maps.IdentityMap(nP=self._nC_residual)
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: maps.IdentityMap):
        if not isinstance(mapping, maps.IdentityMap):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )

        self._mapping = mapping

        for fct in self.objfcts:
            fct.mapping = mapping

def validate_array_type(attribute, array, dtype):
    """Generic array and type validator"""
    if array is not None and not isinstance(array, np.ndarray) and not array.dtype == dtype:
        TypeError(
            f"{attribute} must by a {np.ndarray} of type {dtype}. "
            f"Values of type {type(array)} provided."
        )


def validate_shape(attribute, values, shape):
    """Generic array shape validator"""
    if (values is not None
        and shape != "*"
        and len(values) != shape
    ):
        raise ValueError(
            f"{attribute} must be length {shape} not {len(values)}"
        )


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

    _wire_map: maps.Wires = None

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        # do this as a hack to make TreeMesh work.
        self.regularization_mesh.regularization_type = "Tikhonov"

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

    def __call__(self, model):
        """ Returns the computed value of the coupling term. """
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

