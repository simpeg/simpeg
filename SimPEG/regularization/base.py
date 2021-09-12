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
    _fixed_weights = None
    _free_weights = None
    _mapping = None
    _reference_model = None
    _regularization_mesh = None
    counter = None

    def __init__(self, mesh=None, **kwargs):
        super().__init__()
        self.regularization_mesh = RegularizationMesh(mesh)
        if "active_cells" in kwargs.keys():
            self.regularization_mesh.active_cells = kwargs.pop("active_cells")
        utils.setKwargs(self, **kwargs)

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
    def fixed_weights(self):
        """Regularization weights applied at cell centers"""
        return self._fixed_weights

    @fixed_weights.setter
    def fixed_weights(self, values: np.ndarray):
        self.validate_array_type("fixed_weights", values, float)
        self.validate_shape("fixed_weights", values, self._nC_residual)
        self._fixed_weights = values

    @property
    def free_weights(self):
        """Regularization weights applied at cell centers"""
        return self._free_weights

    @free_weights.setter
    def free_weights(self, values: np.ndarray):
        self.validate_array_type("free_weights", values, float)
        self.validate_shape("free_weights", values, self._nC_residual)
        self._free_weights = values

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
    def W(self):
        """
        Weighting matrix
        """
        raise AttributeError("Regularization class must have a 'W' implementation.")

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
            return mD.T * ((self.W.T * self.W) * mD)

        return mD.T * (self.W.T * (self.W * (mD * v)))

    @staticmethod
    def validate_array_type(attribute, array, dtype):
        """Generic array and type validator"""
        if not isinstance(array, np.ndarray) and not array.dtype == dtype:
            TypeError(
                f"{attribute} must by a {np.ndarray} of type {dtype}. "
                f"Values of type {type(array)} provided."
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
    :param numpy.ndarray fixed_weights: cell weights

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

        if self.fixed_weights is not None:
            weights *= self.fixed_weights

        if self.free_weights is not None:
            weights *= self.free_weights

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
    :param numpy.ndarray fixed_weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray fixed_weights: vector of cell weights (applied in all terms)
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
            self._cell_difference = utils.sdiag(self.length_scales) * getattr(
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
        return "alpha_{orientation}".format(orientation=self.orientation)

    @property
    def W(self):
        """
        Weighting matrix that takes the first spatial difference
        with length scales in the specified orientation.
        """
        average_cell_face = getattr(self.regularization_mesh, "aveCC2F{}".format(self.orientation))
        weights = self.scale * self.regularization_mesh.vol

        if self.fixed_weights is not None:
            weights *= self.fixed_weights

        if self.free_weights is not None:
            weights *= self.free_weights

        return utils.sdiag((average_cell_face * weights) ** 0.5) * self.cell_difference

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
    def length_scales(self, values):
        self.validate_array_type("length_scales", values, float)
        self.validate_shape("length_scales", values, self._nC_residual)
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

###############################################################################
#                                                                             #
#                        Base Combo Regularization                            #
#                                                                             #
###############################################################################


class ComboRegularization(ComboObjectiveFunction):
    _alpha_s = 1.
    _alpha_x = 1.
    _alpha_y = 1.
    _alpha_z = 1.
    _alpha_xx = 1.
    _alpha_yy = 1.
    _alpha_zz = 1.

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
            "fixed_weights",
            "mapping"
        ]

        for attr in linkattrs:
            val = getattr(self, attr)
            if val is not None:
                [setattr(fct, attr, val) for fct in self.objfcts]

    # Properties

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

    reference_model = props.Array("reference model")
    reference_model_in_smooth = properties.Bool(
        "include reference_model in the smoothness calculation?", default=False
    )
    active_cells = properties.Array(
        "indices of active cells in the mesh", dtype=(bool, int)
    )
    regularization_mesh = properties.Instance(
        "regularization mesh", RegularizationMesh, required=True
    )
    mapping = properties.Instance(
        "mapping which is applied to model in the regularization",
        maps.IdentityMap,
        default=maps.IdentityMap(),
    )

    @property
    def fixed_weights(self):
        """Regularization weights applied at cell centers"""
        return self._fixed_weights

    @fixed_weights.setter
    def fixed_weights(self, value):
        for fct in self.objfcts:
            fct.fixed_weights = value

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


class L2Regularization(ComboRegularization):
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
    :param numpy.ndarray fixed_weights: cell weights
    :param bool reference_model_in_smooth: include the reference model in the smoothness computation? (eg. look at Deriv of m (False) or Deriv of (m-reference_model) (True))
    :param numpy.ndarray fixed_weights: vector of cell weights (applied in all terms)

    **Weighting Parameters**

    :param float alpha_s: weighting on the smallness (default 1.)
    :param float alpha_x: weighting on the x-smoothness (default 1.)
    :param float alpha_y: weighting on the y-smoothness (default 1.)
    :param float alpha_z: weighting on the z-smoothness(default 1.)

    """
    def __init__(
        self, mesh, alpha_s=1e-4, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs
    ):

        if "normalized_gradients" not in kwargs:
            kwargs["normalized_gradients"] = False

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
            **kwargs
        )