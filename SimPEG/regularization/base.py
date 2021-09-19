from __future__ import annotations

import numpy as np

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
    _free_weights = None
    _mapping = None
    _reference_model = None
    _regularization_mesh = None
    counter = None
    _free_multiplier = 1.
    _model = None
    _model_units = None
    _W = None

    def __init__(self, mesh=None, **kwargs):
        super().__init__()
        self.regularization_mesh = RegularizationMesh(mesh)
        if "active_cells" in kwargs.keys():
            self.regularization_mesh.active_cells = kwargs.pop("active_cells")
        utils.setKwargs(self, **kwargs)

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
            self.regularization_mesh.active_cells = utils.mkvc(values)

        self._active_cells = values

    @property
    def cell_weights(self):
        """Regularization weights applied at cell centers"""
        return self._cell_weights

    @cell_weights.setter
    def cell_weights(self, values: np.ndarray):
        validate_array_type("cell_weights", values, float)
        validate_shape("cell_weights", values, self._nC_residual)
        self._cell_weights = values

    @property
    def free_multiplier(self):
        return self._free_multiplier

    @free_multiplier.setter
    def free_multiplier(self, value: float):
        validate_array_type("free_weights", value, float)
        if value < 0.:
            raise ValueError("Input free_multiplier must be > 0")

        self._free_multiplier = value

    @property
    def free_weights(self):
        """Regularization weights applied at cell centers"""
        return self._free_weights

    @free_weights.setter
    def free_weights(self, values: np.ndarray | None):

        if values is None:
            self._free_weights = None
        else:
            validate_array_type("free_weights", values, float)
            validate_shape("free_weights", values, self._nC_residual)
            self._free_weights = values

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
        if not isinstance(mapping, maps.IdentityMap):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )

        self._mapping = mapping

    @property
    def model_units(self) -> maps.IdentityMap:
        """Specify the model model_units. Special care given to 'radian' values"""
        return self._model_units

    @model_units.setter
    def model_units(self, model_units: str | None):
        if model_units is not None and not isinstance(model_units, str):
            raise TypeError(
                f"'model_units' must be None or type str. "
                f"Value of type {type(model_units)} provided."
            )
        self._model_units = model_units

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

        if nC != "*" and nC is not None:
            return self.regularization_mesh.nC
        elif mapping is not None and mapping.shape[0] != "*":
            return self.mapping.shape[0]
        else:
            return self.nP

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

            R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top
                   W(m-m_\\text{ref})}

        So the derivative is straight forward:

        .. math::

            R(m) = \mathbf{W^\\top W (m-m_\\text{ref})}

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
    :param int nP: number of parameters
    :param IdentityMap mapping: regularization mapping, takes the model from model space to the space you want to regularize in
    :param numpy.ndarray reference_model: reference model
    :param numpy.ndarray active_cells: active cell indices for reducing the size of differential operators in the definition of a regularization mesh
    :param numpy.ndarray cell_weights: cell weights

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh=None, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

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
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = self.free_multiplier * self.regularization_mesh.vol

            if self.cell_weights is not None:
                weights *= self.cell_weights

            free_weights = self.free_weights
            if free_weights is not None:
                weights *= free_weights

            self._W = utils.sdiag(weights ** 0.5)

        return self._W


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
    _cell_difference = None
    _length_scales = None
    _normalized_gradients: bool = True
    _reference_model_in_smooth: bool = False

    def __init__(self, mesh, orientation="x", **kwargs):
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

        if self.model_units == "radian":
            return utils.mat_utils.coterminal(dfm_dl)
        return dfm_dl

    def f_m_deriv(self, m):
        """
        Derivative of the model gradient
        """
        return self.cell_difference @ self.mapping.deriv(self._delta_m(m))

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
            weights = self.free_multiplier * self.regularization_mesh.vol

            if self.cell_weights is not None:
                weights *= self.cell_weights

            weights = average_cell_2_face * weights

            free_weights = self.free_weights
            if free_weights is not None:
                if len(free_weights) == average_cell_2_face.shape[0]:  # Face weights
                    weights *= free_weights
                else:
                    weights *= average_cell_2_face * free_weights

            self._W = utils.sdiag(self.length_scales * weights ** 0.5)
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
            length_scales = Ave * (
                    self.regularization_mesh.Pac.T * self.regularization_mesh.mesh.h_gridded[:, index]
            )

            if self.normalized_gradients:
                length_scales /= length_scales.min()

            self._length_scales = length_scales**-1.0

        return self._length_scales

    @length_scales.setter
    def length_scales(self, values):
        validate_array_type("length_scales", values, float)
        validate_shape("length_scales", values, self._nC_residual)
        self._length_scales = values

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
    :param numpy.ndarray cell_weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray cell_weights: vector of cell weights (applied in all terms)
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

        if self.model_units == "radian":
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
        weights = self.free_multiplier * self.regularization_mesh.vol

        if self.cell_weights is not None:
            weights *= self.cell_weights

        if self.free_weights is not None:
            weights *= self.free_weights

        return utils.sdiag(weights ** 0.5)

    @property
    def _multiplier_pair(self):
        return f"alpha_{self.orientation}{self.orientation}"

###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################


class BaseComboRegularization(ComboObjectiveFunction):
    _active_cells = None
    _alpha_s = 1.
    _alpha_x = 1.
    _alpha_y = 1.
    _alpha_z = 1.
    _alpha_xx = 1.
    _alpha_yy = 1.
    _alpha_zz = 1.
    _cell_weights = None
    _free_weights = None
    _free_multipliers = None
    _normalized_gradients = True
    _reference_model_in_smooth = False
    _reference_model = None

    def __init__(self, mesh, objfcts=[], **kwargs):

        super().__init__(
            objfcts=objfcts, multipliers=None
        )
        self.regularization_mesh = RegularizationMesh(mesh)
        utils.setKwargs(self, **kwargs)

    @property
    def alpha_s(self):
        """smallness weight"""
        return self._alpha_s

    @alpha_s.setter
    def alpha_s(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_s' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_s = value

    @property
    def alpha_x(self):
        """weight for the first x-derivative"""
        return self._alpha_x

    @alpha_x.setter
    def alpha_x(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_x' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_x = value

    @property
    def alpha_y(self):
        """weight for the first y-derivative"""
        return self._alpha_y

    @alpha_y.setter
    def alpha_y(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_y' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_y = value

    @property
    def alpha_z(self):
        """weight for the first z-derivative"""
        return self._alpha_z

    @alpha_z.setter
    def alpha_z(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_z' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_z = value

    @property
    def alpha_xx(self):
        """weight for the second x-derivative"""
        return self._alpha_xx

    @alpha_xx.setter
    def alpha_xx(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_xx' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_xx = value

    @property
    def alpha_yy(self):
        """weight for the second y-derivative"""
        return self._alpha_yy

    @alpha_yy.setter
    def alpha_yy(self, value):
        if not isinstance(value, float) and value > 0:
            raise ValueError("Input 'alpha_yy' value must me of type float > 0"
                             f"Value {value} of type {type(value)} provided")
        self._alpha_yy = value

    @property
    def alpha_zz(self):
        """weight for the second z-derivative"""
        return self._alpha_zz

    @alpha_zz.setter
    def alpha_zz(self, value):
        if not isinstance(value, float) and value > 0:
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
    def cell_weights(self):
        """Fixed regularization weights applied at cell centers"""
        return self._cell_weights

    @cell_weights.setter
    def cell_weights(self, value: np.ndarray):
        validate_array_type("cell_weights", value, float)
        validate_shape("cell_weights", value, self._nC_residual)
        for fct in self.objfcts:
            fct.cell_weights = value

    @property
    def free_weights(self):
        """Free regularization weights applied at cell centers"""
        return self._free_weights

    @free_weights.setter
    def free_weights(self, value):
        if value is None:
            self._free_weights = None
        else:
            validate_array_type("free_weights", value, float)
            validate_shape("free_weights", value, self._nC_residual)
            self._free_weights = value

    @property
    def free_multipliers(self):
        """Free regularization multipliers applied at cell centers"""
        return self._free_multipliers

    @free_multipliers.setter
    def free_multipliers(self, values: np.ndarray | list):
        if len(values) != len(self.objfcts):
            raise ValueError(
                f"List of 'free_multipliers' must be of length {len(self.objfcts)}. "
                f"List or array of length {len(values)} provided."
            )
        for fct, value in zip(self.objfcts, values):
            fct.free_multiplier = value

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
        mapping = getattr(self, "_mapping", None)

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
        validate_shape("active_cells", values, self._nC_residual)

        if getattr(self, "regularization_mesh", None) is not None:
            self.regularization_mesh.active_cells = utils.mkvc(values)

        self._active_cells = values

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

    @property
    def model_units(self) -> maps.IdentityMap:
        """Specify the model model_units. Special care given to 'radian' values"""
        return self._model_units

    @model_units.setter
    def model_units(self, model_units: str | None):
        if model_units is not None and not isinstance(model_units, str):
            raise TypeError(
                f"'model_units' must be None or type str. "
                f"Value of type {type(model_units)} provided."
            )
        for fct in self.objfcts:
            fct.model = model_units

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
        for fct in self.objfcts:
            fct.regularization_mesh = mesh

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


class L2Regularization(BaseComboRegularization):
    """
    Simple regularization that measures the l2-norm of the model and model gradients.

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
        self,
        mesh,
        alpha_s=1e-4,
        alpha_x=1.0,
        alpha_y=1.0,
        alpha_z=1.0,
        normalized_gradients=False,
        **kwargs
    ):
        objfcts = [
            Small(mesh=mesh, **kwargs),
            SmoothDeriv(mesh=mesh, orientation="x", **kwargs),
        ]

        if mesh.dim > 1:
            objfcts.append(SmoothDeriv(mesh=mesh, orientation="y", **kwargs))

        if mesh.dim > 2:
            objfcts.append(SmoothDeriv(mesh=mesh, orientation="z", **kwargs))

        super().__init__(
            mesh=mesh,
            objfcts=objfcts,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            normalized_gradients=normalized_gradients,
            **kwargs
        )


def validate_array_type(attribute, array, dtype):
    """Generic array and type validator"""
    if not isinstance(array, np.ndarray) and not array.dtype == dtype:
        TypeError(
            f"{attribute} must by a {np.ndarray} of type {dtype}. "
            f"Values of type {type(array)} provided."
        )


def validate_shape(attribute, values, shape):
    """Generic array shape validator"""
    if (
        shape != "*"
        and len(values) != shape
    ):
        raise ValueError(
            f"{attribute} must be length {shape} not {len(values)}"
        )