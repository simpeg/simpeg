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
    """Base class for least-squares regularization.

    The ``BaseRegularization`` class defines properties and methods inherited by least-squares
    regularization classes. It is not directly used to constrain the inversions.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is defined. This is not necessarily the same as the mesh on which the simulation is defined.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~SimPEG.regularization.RegularizationMesh` cells that are active in the inversion.
        If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

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
        """Active cells defined on the regularization mesh.

        A boolean array defining the cells in the :py:class:`~SimPEG.regularization.RegularizationMesh`
        that are active throughout the inversion. Inactive cells remain fixed and are defined according
        to the starting model.

        Returns
        -------
        (n_cells, ) Array of bool

        Notes
        -----
        If the property is set using an array of integers, the setter interprets the array as
        representing the indices of the active cells. When called however, the quantity will have
        been converted to a boolean array.
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
        "active_cells",
        "0.19.0",
        future_warn=True,
        error=False,
    )

    @property
    def model(self) -> np.ndarray:
        """The model associated with regularization.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The model parameters.
        """
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
        """Mapping from the model to the regularization mesh.

        Returns
        -------
        SimPEG.maps.BaseMap
            The mapping from the model parameters to the quantity defined on the :py:class:`~SimPEG.regularization.RegularizationMesh`.
        """
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
        """Units for the model parameters.

        Some regularization classes behave differently depending on the units; e.g. 'radian'.

        Returns
        -------
        str
            Units for the model parameters.
        """
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
        """Reference model.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Reference model.
        """
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
        "reference_model",
        "0.19.0",
        future_warn=True,
        error=False,
    )

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh.

        Mesh on which the regularization is defined. This is not the same as the mesh on which the simulation is defined.

        Returns
        -------
        discretize.base.RegularizationMesh
            Mesh on which the regularization is defined.
        """
        return self._regularization_mesh

    regmesh = deprecate_property(
        regularization_mesh,
        "regmesh",
        "regularization_mesh",
        "0.19.0",
        future_warn=True,
        error=False,
    )

    @property
    def cell_weights(self) -> np.ndarray:
        """Deprecated property for 'volume' and user defined weights."""
        warnings.warn(
            "cell_weights are deprecated please access weights using the `set_weights`,"
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.19.0",
            FutureWarning,
        )
        return np.prod(list(self._weights.values()), axis=0)

    @cell_weights.setter
    def cell_weights(self, value):
        warnings.warn(
            "cell_weights are deprecated please access weights using the `set_weights`,"
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.19.0",
            FutureWarning,
        )
        self.set_weights(cell_weights=value)

    def get_weights(self, key) -> np.ndarray:
        """Cell weights for a given key.

        Returns
        -------
        (n_cells, ) numpy.ndarray
            Cell weights for a given key.

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
        return self._weights[key]

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights to the regularization

        Parameters
        ----------
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
        """Removes the weights for the key provided

        Parameters
        ----------
        key : str
            The key for the weights being removed from the cell weights dictionary.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> reg.remove_weights('my_weight')
        """
        try:
            self._weights.pop(key)
        except KeyError as error:
            raise KeyError(f"{key} is not in the weights dictionary") from error
        self._W = None

    @property
    def W(self) -> np.ndarray:
        r"""Weighting matrix.

        For the set of (n_cells, ) numpy arrays :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}`
        defining cell weights applied in the regularization, this method returns the cell weighting
        matrix :math:`\mathbf{W}`, where:

        .. math::
            \mathbf{W} = diag \bigg ( \prod_i \, \mathbf{w_i} \bigg )

        In this case, the product represents elementwise multiplications; i.e. the Hadamard product.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the regularization.
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = utils.sdiag(weights**0.5)
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
        r"""
        We use a weighted 2-norm objective function

        .. math::

            r(m) = \frac{1}{2} \| \mathbf{W} \mathbf{f(m)} \|_2^2
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
        r"""Gradient of the regularization function for the model provided.

        Where :math:`\phi_m` represents the regularization function,
        this method returns the gradient:

        .. math::
            \nabla_m \phi_m = \frac{\partial \phi_m}{\partial \mathbf{m}} \bigg |_\mathbf{m}

        evaluated at the model :math:`(\mathbf{m})` provided.

        Parameters
        ----------
        (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The gradient of the regularization function evaluated for the model provided.

        """
        r = self.W * self.f_m(m)
        return self.f_m_deriv(m).T * (self.W.T * r)

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        r"""Second derivative of the regularization function.

        Where :math:`\phi_m` represents the regularization function,
        this method returns either the second derivative evaluated at the model :math:`(\mathbf{m})` provided

        .. math::
            \nabla_m^2 \phi_m = \frac{\partial^2 \phi_m}{\partial \mathbf{m}^2} \bigg |_\mathbf{m}

        or the second-derivative multiplied by a given vector :math:`(\mathbf{v})`

        .. math::
            \big [ \nabla_m^2 \phi_m \big ] \mathbf{v} = \bigg [ \frac{\partial^2 \phi_m}{\partial \mathbf{m}^2} \bigg |_\mathbf{m} \bigg ] \mathbf{v}


        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.
        v : None, (n_param, ) numpy.ndarray (optional)
            A vector

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix | (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the second-derivative of the regularization function for the model
            provided is returned. If *v* is not ``None``, the second-derivative multiplied by the vector provided is returned.

        """
        f_m_deriv = self.f_m_deriv(m)
        if v is None:
            return f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)

        return f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))


class Smallness(BaseRegularization):
    r"""Smallness least-squares regularization.

    Smallness least-squares regularization is used to ensure recovered model parameter
    values are not overly large in amplitude. In continuous form, the ``Smallness``
    regularization function is given by:

    .. math::
        r(m) = \frac{1}{2} \int_\Omega \, w(r) \, m(r)^2 \, dv

    where :math:`w(r)` is a user-defined weighting function.
    In discrete form, the ``Smallness`` regularization function is given by:

    .. math::
        r(\mathbf{m}) = \frac{1}{2} \big ( \mathbf{m - m_{ref}})^T \mathbf{V^T}
        \mathbf{W}^T \mathbf{W} \mathbf{V} (\mathbf{m} - \mathbf{m_{ref}})

    where

        - :math:`\mathbf{m}` is the model,
        - :math:`\mathbf{m_{ref}}` is a reference model,
        - :math:`\mathbf{V}` is a diagonal matrix containing the **square roots** of cell volumes and
        - :math:`\mathbf{W}` is a weighting matrix (default Identity). If fixed or free weights are provided, then :math:`\mathbf{W}` contains the **square roots** of the weights provided.


    Parameters
    ----------
    mesh : discretize.base.BaseMesh mesh
        The mesh on which the regularization is defined
    shape : int
        The number of model parameters
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)

    def f_m(self, m) -> np.ndarray:
        r"""Evaluate least squares regularization kernel.

        For ``Smallness`` regularization, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m - m_{ref}}

        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel.

        Notes
        -----
        The discretized form of the smallness regularization function is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{W^T W} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. The regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m - m_{ref}}

        where :math:`\mathbf{m_{ref}}` is a reference model.
        """
        return self.mapping * self._delta_m(m)

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the least-squares regularization kernel.

        For ``Smallness`` regularization, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m - m_{ref}}

        And thus, the derivative with respect to the model is the identity matrix :math:`\mathbf{I}`.
        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the least-squares regularization kernel.

        Notes
        -----
        The discretized form of the smallness regularization function function is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{W^T W} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. The regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m - m_{ref}}

        where :math:`\mathbf{m_{ref}}` is a reference model. Thus, the derivate with respect to the
        model is:

        .. math::
            \frac{\partial \partial{f_m}}{\partial \mathbf{m}} = \mathbf{I}
        """
        return self.mapping.deriv(self._delta_m(m))


class SmoothnessFirstOrder(BaseRegularization):
    r"""First-order smoothness least-squares regularization.

    First-order smoothness least-squares regularization is used to enforce spatial smoothness
    in the recovered model along a specified direction. The regularization accomplishes this by
    penalizing models with large first-order spatial derivatives along a specified direction.

    For a ``SmoothnessFirstOrder`` regularization that enforces smoothness along the x-direction,
    the continuous regularization function is given by:

    .. math::
        r(m) = \frac{1}{2} \int_\Omega \, w (r) \bigg ( \frac{\partial m}{\partial x} \bigg )^2 \, dv

    where :math:`w(r)` is a user-defined weighting function.
    In discrete form, this regularization function is given by:

    .. math::
        r(\mathbf{m}) = \frac{1}{2} \big ( \mathbf{m - m_{ref}})^T \mathbf{G_x^T V}^T}
        \mathbf{W}^T \mathbf{W} \mathbf{V} G_x } (\mathbf{m} - \mathbf{m_{ref}})

    where

        - :math:`\mathbf{m}` is the model,
        - :math:`\mathbf{m_{ref}}` is a reference model (optional),
        - :math:`\mathbf{G_x}` is the partial gradient operator along the x-direction (i.e. x-derivative),
        - :math:`\mathbf{V}` is a diagonal matrix containing the **square roots** of cell volumes and
        - :math:`\mathbf{W}` is a weighting matrix (default Identity). If fixed or free weights are provided, then :math:`\mathbf{W}` contains the **square roots** of the weights provided.


    Parameters
    ----------
    mesh : discretize.base.BaseMesh mesh
        The mesh on which the regularization is defined.
    orientation : str {'x', 'y', 'z'}
        The direction along which smoothness is enforced. Default = 'x'.
    reference_model_in_smooth : bool
        Whether the reference model is included in the smoothness regularization. Default = ``False``.
    shape : int
        The number of model parameters
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

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
        """Acceptable lengths for the weights.

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
        """Partial cell gradient operator.

        Returns the partial gradient operator which takes the derivative along the orientation
        where smoothness is being enforced. For smoothness along the x-direction, the resulting operator would
        map from cell centers to x-faces.

        Returns
        -------
        scipy.sparse.csr_matrix
            Partial cell gradient operator defined on the :py:class:`~SimPEG.regularization.RegularizationMesh`.
        """
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
        r"""Evaluate least squares regularization kernel.

        For ``SmoothnessFirstOrder`` regularization in the x-direction, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel.

        Notes
        -----
        The discretized form of the smoothness regularization function in along the x-direction is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{G_x^T W^T W G_x} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. Thus the regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{m_{ref}}` is a reference model.
        """
        dfm_dl = self.cell_gradient @ (self.mapping * self._delta_m(m))

        if self.units is not None and self.units.lower() == "radian":
            return (
                utils.mat_utils.coterminal(dfm_dl * self._cell_distances)
                / self._cell_distances
            )
        return dfm_dl

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the least-squares regularization kernel.

        For ``SmoothnessFirstOrder`` regularization, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        And thus, the derivative with respect to the model is the x-derivative operator :math:`\mathbf{G_x}`.
        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the least-squares regularization kernel.

        Notes
        -----
        The discretized form of the smoothness regularization function in along the x-direction is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{G_x^T W^T W G_x} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. Thus the regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{m_{ref}}` is a reference model. Thus, the derivate with respect to the
        model is:

        .. math::
            \frac{\partial \partial{f_m}}{\partial \mathbf{m}} = \mathbf{G_x}
        """
        return self.cell_gradient @ self.mapping.deriv(self._delta_m(m))

    @property
    def W(self):
        """Weighting matrix.

        A sparse, diagonal weighting matrix for all weights associated with the
        regularization object. This includes default weights that are set when the regularization object
        is instantiated (e.g. cell volumes and length scales corresponding to the
        difference operator), as well as any user-defined weights.

        Returns
        -------
        scipy.sparse.csr_matrix
            Weighting matrix.
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
            self._W = utils.sdiag(weights**0.5)
        return self._W

    @property
    def _cell_distances(self):
        """
        Distances between cell centers for the cell center difference.
        """
        return getattr(self.regularization_mesh, f"cell_distances_{self.orientation}")

    @property
    def orientation(self):
        """Direction along which smoothness is enforced.

        Returns
        -------
        str {'x', 'y', 'z'}
            The direction along which smoothness is enforced. Default = 'x'.

        """
        return self._orientation


class SmoothnessSecondOrder(SmoothnessFirstOrder):
    r"""Second-order smoothness (flatness) least-squares regularization.

    Second-order smoothness least-squares regularization is used to enforce flatness
    in the recovered model along a specified direction. The regularization accomplishes this by
    penalizing models with large second-order spatial derivatives along a specified direction.

    For a ``SmoothnessSecondOrder`` regularization that enforces flatness along the x-direction,
    the continuous regularization function is given by:

    .. math::
        r(m) = \frac{1}{2} \int_\Omega \, w (r) \bigg ( \frac{\partial^2 m}{\partial x^2} \bigg )^2 \, dv

    where :math:`w(r)` is a user-defined weighting function.
    In discrete form, this regularization function is given by:

    .. math::
        r(\mathbf{m}) = \frac{1}{2} \big ( \mathbf{m - m_{ref}})^T \mathbf{L_x^T V}^T}
        \mathbf{W}^T \mathbf{W} \mathbf{V} L_x } (\mathbf{m} - \mathbf{m_{ref}})

    where

        - :math:`\mathbf{m}` is the model,
        - :math:`\mathbf{m_{ref}}` is a reference model (optional),
        - :math:`\mathbf{L_x}` is the second-order scalar derivative with respect to x,
        - :math:`\mathbf{V}` is a diagonal matrix containing the **square roots** of cell volumes and
        - :math:`\mathbf{W}` is a weighting matrix (default Identity). If fixed or free weights are provided, then :math:`\mathbf{W}` contains the **square roots** of the weights provided.


    Parameters
    ----------
    mesh : discretize.base.BaseMesh mesh
        The mesh on which the regularization is defined.
    orientation : str {'x', 'y', 'z'}
        The direction along which smoothness is enforced. Default = 'x'.
    reference_model_in_smooth : bool
        Whether the reference model is included in the smoothness regularization. Default = ``False``.
    shape : int
        The number of model parameters
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, ) numpy.ndarray that is defined on
        the :py:class:`~SimPEG.regularization.RegularizationMesh`.

    """

    def f_m(self, m):
        r"""Evaluate least squares regularization kernel.

        For ``SmoothnessSecondOrder`` regularization in the x-direction, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x^T} \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel.

        Notes
        -----
        The discretized form of the smoothness regularization function in along the x-direction is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{G_x G_x^T W^T W G_x^T G_x} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. Thus the regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x^T G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{m_{ref}}` is a reference model.
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
        r"""Derivative of the least-squares regularization kernel.

        For ``SmoothnessSecondOrder`` regularization, the regularization kernel is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x^T G_x} \big ( \mathbf{m - m_{ref}} \big )

        And thus, the derivative with respect to the model is :math:`\mathbf{G_x^T G_x}`.
        For a more detailed, description, see the notes.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the least-squares regularization kernel.

        Notes
        -----
        The discretized form of the smoothness regularization function in along the x-direction is expressed as:

        .. math::
            \begin{align}
            \phi_m (\mathbf{m}) &= \frac{1}{2} \big ( \mathbf{m - m_{ref}} \big )^T \mathbf{G_x G_x^T W^T W G_x^T G_x} \big ( \mathbf{m - m_{ref}} \big ) \\
            &= \frac{1}{2} \mathbf{f_m}^T \mathbf{W^T W} \, \mathbf{f_m}
            \end{align}

        where :math:`\mathbf{G_x}` is the discrete x-derivative operator.
        :math:`\mathbf{W}` is a weighting matrix that applies user-defined weights and accounts
        for cell dimensions in the integration. Thus the regularization kernel :math:`\mathbf{f_m}` is defined as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big ( \mathbf{m - m_{ref}} \big )

        where :math:`\mathbf{m_{ref}}` is a reference model. Thus, the derivate with respect to the
        model is:

        .. math::
            \frac{\partial \partial{f_m}}{\partial \mathbf{m}} = \mathbf{G_x^T G_x}
        """
        return (
            self.cell_gradient.T
            @ self.cell_gradient
            @ self.mapping.deriv(self._delta_m(m))
        )

    @property
    def W(self):
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = utils.sdiag(weights**0.5)

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
    r"""Weighted least squares smallness and smoothness regularization.

    Construct a regularization using a weighted sum of smallness and smoothness
    least-squares regularization functions.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh
        The mesh on which the model parameters are defined. This is used
        for constructing difference operators for the smoothness terms.
    active_cells : array_like of bool or int, optional
        List of active cell indices, or a `mesh.n_cells` boolean array
        describing active cells.
    alpha_s : float, optional
        Smallness weight
    alpha_x, alpha_y, alpha_z : float or None, optional
        First order smoothness weights for the respective dimensions.
        `None` implies setting these weights using the `length_scale`
        parameters.
    alpha_xx, alpha_yy, alpha_zz : 0, float
        Second order smoothness weights for the respective dimensions.
    length_scale_x, length_scale_y, length_scale_z : float, optional
        First order smoothness length scales for the respective dimensions.
    mapping : SimPEG.maps.IdentityMap, optional
        A mapping to apply to the model before regularization.
    reference_model : array_like, optional
        Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness terms.
    weights : None, array_like, or dict or array_like, optional
        User defined weights. It is recommended to interact with weights using
        the `get_weights`, `set_weights` functionality.

    Notes
    -----
    The regularization function approximates:

    .. math::
        \phi_m(\mathbf{m}) &= \alpha_s \| W_s (\mathbf{m} - \mathbf{m_{ref}} ) \|^2 \\
        & + \alpha_x \| W_x \frac{\partial}{\partial x} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \alpha_y \| W_y \frac{\partial}{\partial y} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \alpha_z \| W_z \frac{\partial}{\partial z} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2 \\
        & + \alpha_{xx} \| W_{xx} \frac{\partial^2}{\partial x^2} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \alpha_{yy} \| W_{yy} \frac{\partial^2}{\partial y^2} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2
        + \alpha_{zz} \| W_{zz} \frac{\partial^2}{\partial z^2} (\mathbf{m} - \mathbf{m_{ref}} ) \|^2

    By default, :math:`\alpha_{xx}=\alpha_{yy}=\alpha_{zz}=0`.
    Note if the key word argument `reference_model_in_smooth` is False, then :math:`m_{ref}` is not
    included in the smoothness contribution.

    If length scales are used to set the smoothness weights, alphas are respectively set internally using:
    >>> alpha_x = (length_scale_x * min(mesh.edge_lengths)) ** 2
    """

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
        """Adds (or updates) the specified weights for all child regularization objects.

        Parameters
        ----------
        **kwargs : key, numpy.ndarray
            Each keyword argument is added to the weights used by all child regularization objects.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import WeightedLeastSquares
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = WeightedLeastSquares(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        """
        for fct in self.objfcts:
            fct.set_weights(**weights)

    def remove_weights(self, key):
        """Removes specified weights from all child regularization objects.

        Parameters
        ----------
        key : str
            The key for the weights being removed from all child regularization objects.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import WeightedLeastSquares
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = WeightedLeastSquares(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> reg.remove_weights('my_weight')
        """
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
            " `get_weights`, and `remove_weights` functionality. This will be removed in 0.19.0",
            FutureWarning,
        )
        self.set_weights(cell_weights=value)

    @property
    def alpha_s(self):
        """Weighting constant for smallness term.

        Returns
        -------
        float
            Weighting constant for smallness term.
        """
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
        """Weighting constant for first order x-derivative term.

        Returns
        -------
        float
            Weighting constant for first order x-derivative term.
        """
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
        """Weighting constant for first order y-derivative term.

        Returns
        -------
        float
            Weighting constant for first order y-derivative term.
        """
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
        """Weighting constant for first order z-derivative term.

        Returns
        -------
        float
            Weighting constant for first order z-derivative term.
        """
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
        """Weighting constant for second order x-derivative term.

        Returns
        -------
        float
            Weighting constant for second order x-derivative term.
        """
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
        """Weighting constant for second order y-derivative term.

        Returns
        -------
        float
            Weighting constant for second order y-derivative term.
        """
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
        """Weighting constant for second order z-derivative term.

        Returns
        -------
        float
            Weighting constant for second order z-derivative term.
        """
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
        """Constant multiplier of the base length scale on model gradients along x.

        Returns
        -------
        float
            Constant multiplier of the base length scale on model gradients along x.
        """
        return np.sqrt(self.alpha_x) / self.regularization_mesh.base_length

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
        """Constant multiplier of the base length scale on model gradients along y.

        Returns
        -------
        float
            Constant multiplier of the base length scale on model gradients along y.
        """
        return np.sqrt(self.alpha_y) / self.regularization_mesh.base_length

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
        """Constant multiplier of the base length scale on model gradients along z.

        Returns
        -------
        float
            Constant multiplier of the base length scale on model gradients along z.
        """
        return np.sqrt(self.alpha_z) / self.regularization_mesh.base_length

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
        """Whether to include the reference model in the smoothness terms.

        Returns
        -------
        bool
            Whether to include the reference model in the smoothness terms.
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
        """Number of model parameters.

        Returns
        -------
        int
            Number of model parameters.
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
        """Multipliers for weighted sum of objective functions.

        Returns
        -------
        list of float
            Multipliers for weighted sum of objective functions.
        """
        return [getattr(self, objfct._multiplier_pair) for objfct in self.objfcts]

    @property
    def active_cells(self) -> np.ndarray:
        """Active cells defined on the regularization mesh.

        A boolean array defining the cells in the :py:class:`~SimPEG.regularization.RegularizationMesh`
        that are active throughout the inversion. Inactive cells remain fixed and are defined according
        to the starting model.

        Returns
        -------
        (n_cells, ) Array of bool

        Notes
        -----
        If the property is set using an array of integers, the setter interprets the array as
        representing the indices of the active cells. When called however, the quantity will have
        been converted to a boolean array.
        """
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
        "active_cells",
        "0.19.0",
        error=False,
        future_warn=True,
    )

    @property
    def reference_model(self) -> np.ndarray:
        """Reference model values used to constrain the inversion.

        Returns
        -------
        array_like
            Reference model values used to constrain the inversion. If ``None``, the reference model is equal to the starting model for the inversion.
        """
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
        "reference_model",
        "0.19.0",
        future_warn=True,
        error=False,
    )

    @property
    def model(self) -> np.ndarray:
        """The model associated with regularization.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The model parameters.
        """
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
        """Units for the model parameters.

        Some regularization classes behave differently depending on the units; e.g. 'radian'.

        Returns
        -------
        str
            Units for the model parameters.
        """
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
        """Regularization mesh.

        Mesh on which the regularization is defined. This is not the same as the mesh on which the simulation is defined.

        Returns
        -------
        discretize.base.RegularizationMesh
            Mesh on which the regularization is defined.
        """
        return self._regularization_mesh

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping from the model to the regularization mesh.

        Returns
        -------
        SimPEG.maps.BaseMap
            The mapping from the model parameters to the quantity defined on the :py:class:`~SimPEG.regularization.RegularizationMesh`.
        """
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
    """Base class for the similarity term in joint inversions.

    The ``BaseSimilarityMeasure`` assumes two different geophysical models through one similarity term.
    Inherit this for building your own similarity term.
    However, if you wish to combine more than two models, e.g., 3 models,
    you may want to add a total of three coupling terms:

    e.g., lambda1*(m1, m2) + lambda2*(m1, m3) + lambda3*(m2, m3)

    where, lambdas are weights for coupling terms. m1, m2 and m3 indicate
    three different models.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh
        Mesh on which the regularization is defined. This is not necessarily the same as the mesh on which the simulation is defined.
    mapping : SimPEG.maps.WireMap
        Wire map connecting physical properties defined on the regularization mesh to the entire model.
    """

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, **kwargs)
        self.wire_map = wire_map

    @property
    def wire_map(self):
        """Wire map connecting physical properties to the entire model.

        Returns
        -------
        SimPEG.maps.WireMap
            Wire map connecting physical properties defined on the regularization mesh to the entire model.
        """
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
        """Number of model parameters.

        Returns
        -------
        int
            Number of model parameters.
        """
        return self.wire_map.nP

    def deriv(self, model):
        """First derivative of the coupling term with respect to individual models.

        Where :math:`k` is the number of models we are inverting for and :math:`M` is the number of cells in each model,
        this method returns a vector of length :math:`kM`.

        Parameters
        ----------
        model : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            First derivative of the coupling term with respect to individual models.

        """
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self, model, v=None):
        """Second derivative of the coupling term with respect to individual models.

        Parameters
        ----------
        model : numpy.ndarray
            The model.
        v : numpy.ndarray, optional
            A vector.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            Where :math:`k` is the number of models we are inverting for and :math:`M` is the number of cells in each model,
            this method returns:

                - an array of dimensions (k*M, ) when `v` is not ``None``.
                - a sparse matrix of dimensions (k*M, k*M) when `v` is ``None``.


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
