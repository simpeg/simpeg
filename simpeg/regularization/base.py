from __future__ import annotations

import numpy as np
from discretize.base import BaseMesh
from typing import TYPE_CHECKING
from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import RegularizationMesh

from simpeg.utils.code_utils import deprecate_property, validate_ndarray_with_shape

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseRegularization(BaseObjectiveFunction):
    """Base regularization class.

    The ``BaseRegularization`` class defines properties and methods inherited by
    SimPEG regularization classes, and is not directly used to construct inversions.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active cells in the inversion.
        If ``None``, the mapping is set to :obj:`simpeg.maps.IdentityMap`.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function.
        Each value is a numpy.ndarray of shape(:py:property:`~.regularization.RegularizationMesh.n_cells`, ).

    """

    _model = None
    _parent = None
    _W = None

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
        if weights is not None and not isinstance(weights, dict):
            raise TypeError(
                f"Invalid 'weights' of type '{type(weights)}'. "
                "It must be a dictionary with strings as keys and arrays as values."
            )

        # Raise errors on deprecated arguments: avoid old code that still uses
        # them to silently fail
        if (key := "indActive") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. "
                "Please use 'active_cells' instead."
            )
        if (key := "cell_weights") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. Please use 'weights' instead."
            )

        super().__init__(nP=None, mapping=None, **kwargs)
        self._regularization_mesh = mesh
        self._weights = {}
        if active_cells is not None:
            self.active_cells = active_cells
        self.mapping = mapping  # Set mapping using the setter
        self.reference_model = reference_model
        self.units = units
        if weights is not None:
            self.set_weights(**weights)

    @property
    def active_cells(self) -> np.ndarray:
        """Active cells defined on the regularization mesh.

        A boolean array defining the cells in the :py:class:`~.regularization.RegularizationMesh`
        that are active (i.e. updated) throughout the inversion. The values of inactive cells
        remain equal to their starting model values.

        Returns
        -------
        (n_cells, ) array of bool

        Notes
        -----
        If the property is set using a ``numpy.ndarray`` of ``int``, the setter interprets the
        array as representing the indices of the active cells. When called however, the quantity
        will have been internally converted to a boolean array.
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
        error=True,
    )

    @property
    def model(self) -> np.ndarray:
        """The model parameters.

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
        """Mapping from the inversion model parameters to the regularization mesh.

        Returns
        -------
        simpeg.maps.BaseMap
            The mapping from the inversion model parameters to the quantity defined on the
            :py:class:`~.regularization.RegularizationMesh`.
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
    def parent(self):
        """
        The parent objective function
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        combo_class = ComboObjectiveFunction
        if not isinstance(parent, combo_class):
            raise TypeError(
                f"Invalid parent of type '{parent.__class__.__name__}'. "
                f"Parent must be a {combo_class.__name__}."
            )
        self._parent = parent

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
        None, (n_param, ) numpy.ndarray
            Reference model. If ``None``, the reference model in the inversion is set to
            the starting model.
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
        error=True,
    )

    @property
    def regularization_mesh(self) -> RegularizationMesh:
        """Regularization mesh.

        Mesh on which the regularization is discretized. This is not the same as
        the mesh on which the simulation is defined. See :class:`.regularization.RegularizationMesh`

        Returns
        -------
        .regularization.RegularizationMesh
            Mesh on which the regularization is discretized.
        """
        return self._regularization_mesh

    regmesh = deprecate_property(
        regularization_mesh,
        "regmesh",
        "regularization_mesh",
        "0.19.0",
        error=True,
    )

    @property
    def cell_weights(self) -> np.ndarray:
        """Deprecated property for 'volume' and user defined weights."""
        raise AttributeError(
            "'cell_weights' has been removed. "
            "Please access weights using the `set_weights`, `get_weights`, and "
            "`remove_weights` methods."
        )

    @cell_weights.setter
    def cell_weights(self, value):
        raise AttributeError(
            "'cell_weights' has been removed. "
            "Please access weights using the `set_weights`, `get_weights`, and "
            "`remove_weights` methods."
        )

    def get_weights(self, key) -> np.ndarray:
        """Cell weights for a given key.

        Parameters
        ----------
        key: str
            Name of the weights requested.

        Returns
        -------
        (n_cells, ) numpy.ndarray
            Cell weights for a given key.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        """
        return self._weights[key]

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights to the regularization.

        Parameters
        ----------
        **kwargs : key, numpy.ndarray
            Each keyword argument is added to the weights used by the regularization.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import Smallness
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

    @property
    def weights_keys(self) -> list[str]:
        """
        Return the keys for the existing cell weights
        """
        return list(self._weights.keys())

    def remove_weights(self, key):
        """Removes the weights for the key provided.

        Parameters
        ----------
        key : str
            The key for the weights being removed from the cell weights dictionary.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import Smallness
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
    def W(self) -> csr_matrix:
        r"""Weighting matrix.

        Returns the weighting matrix for the discrete regularization function. To see how the
        weighting matrix is constructed, see the *Notes* section for the :class:`Smallness`
        regularization class.

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
        """Evaluate the regularization function for the model provided.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the function is evaluated.

        Returns
        -------
        float
            The regularization function evaluated for the model provided.
        """
        r = self.W * self.f_m(m)
        return r.dot(r)

    def f_m(self, m) -> np.ndarray:
        """Not implemented for ``BaseRegularization`` class."""
        raise AttributeError("Regularization class must have a 'f_m' implementation.")

    def f_m_deriv(self, m) -> csr_matrix:
        """Not implemented for ``BaseRegularization`` class."""
        raise AttributeError(
            "Regularization class must have a 'f_m_deriv' implementation."
        )

    @utils.timeIt
    def deriv(self, m) -> np.ndarray:
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters; i.e.
        the gradient:

        .. math::
            \frac{\partial \phi}{\partial \mathbf{m}}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            The Gradient of the regularization function evaluated for the model provided.
        """
        r = self.W * self.f_m(m)
        return 2 * self.f_m_deriv(m).T * (self.W.T * r)

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        r"""Hessian of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method returns the second-derivative (Hessian) with respect to the model parameters:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2}

        or the second-derivative (Hessian) multiplied by a vector :math:`(\mathbf{v})`:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the Hessian is evaluated.
        v : None, (n_param, ) numpy.ndarray (optional)
            A vector.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix | (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian of the regularization
            function for the model provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """
        f_m_deriv = self.f_m_deriv(m)
        if v is None:
            return 2 * f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)

        return 2 * f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))


class Smallness(BaseRegularization):
    r"""Smallness regularization for least-squares inversion.

    ``Smallness`` regularization is used to ensure that differences between the
    model values in the recovered model and the reference model are small;
    i.e. it preserves structures in the reference model. If a reference model is not
    supplied, the starting model will be set as the reference model in the
    corresponding objective function by default. Optionally, custom cell weights can be
    included to control the degree of smallness being enforced throughout different
    regions the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : .regularization.RegularizationMesh
        Mesh on which the regularization is discretized. Not the mesh used to
        define the simulation.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active cells in the inversion.
        If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to
        a (n_cells, ) numpy.ndarray that is defined on the
        :py:class:`regularization.RegularizationMesh` .

    Notes
    -----
    We define the regularization function (objective function) for smallness as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \Big [ m(r) - m^{(ref)}(r) \Big ]^2 \, dv

    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model and :math:`w(r)`
    is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \bigg | \, m_i - m_i^{(ref)} \, \bigg |^2

    where :math:`m_i \in \mathbf{m}` are the discrete model parameter values defined on the mesh and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply any user-defined weighting.
    This is equivalent to an objective function of the form:

    .. math::
        \phi (\mathbf{m}) =
        \Big \| \mathbf{W} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    where

        - :math:`\mathbf{m}^{(ref)}` is a reference model (set using `reference_model`), and
        - :math:`\mathbf{W}` is the weighting matrix.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v}` are the cell volumes.
    The weighting matrix used to apply the weights for smallness regularization is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = Smallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2)

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)

    def f_m(self, m) -> np.ndarray:
        r"""Evaluate the regularization kernel function.

        For smallness regularization, the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m} - \mathbf{m}^{(ref)}

        where :math:`\mathbf{m}` are the discrete model parameters and :math:`\mathbf{m}^{(ref)}`
        is a reference model. For a more detailed description, see the *Notes* section below.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function evaluated for the model provided.

        Notes
        -----
        The objective function for smallness regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters defined on the mesh (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`Smallness` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m} - \mathbf{m}^{(ref)}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        """
        return self.mapping * self._delta_m(m)

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the regularization kernel function.

        For ``Smallness`` regularization, the derivative of the regularization kernel function
        with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{I}

        where :math:`\mathbf{I}` is the identity matrix.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the regularization kernel function.

        Notes
        -----
        The objective function for smallness regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters defined on the mesh (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`Smallness` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m} - \mathbf{m}^{(ref)}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        Thus, the derivative with respect to the model is:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{I}

        where :math:`\mathbf{I}` is the identity matrix.
        """
        return self.mapping.deriv(self._delta_m(m))


class SmoothnessFirstOrder(BaseRegularization):
    r"""First-order smoothness least-squares regularization.

    ``SmoothnessFirstOrder`` regularization is used to ensure that values in the recovered
    model are smooth along a specified direction. When a reference model is included,
    the regularization preserves gradients/interfaces within the reference model along
    the direction specified (x, y or z). Optionally, custom cell weights can be used
    to control the degree of smoothness being enforced throughout different regions
    the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh mesh
        The mesh on which the regularization is discretized.
    orientation : {'x', 'y', 'z'}
        The direction along which smoothness is enforced.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active cells in the inversion.
        If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model. To include the reference model in the regularization, the
        `reference_model_in_smooth` property must be set to ``True``.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness regularization.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently
        depending on the units; e.g. 'radian'.
    weights : None, dict
        Custom weights for the least-squares function. Each ``key`` points to
        a ``numpy.ndarray`` that is defined on the :py:class:`regularization.RegularizationMesh`.
        A (n_cells, ) ``numpy.ndarray`` is used to define weights at cell centers, which are
        averaged to the appropriate faces internally when weighting is applied.
        A (n_faces, ) ``numpy.ndarray`` is used to define weights directly on the faces specified
        by the `orientation` input argument.

    Notes
    -----
    We define the regularization function (objective function) for first-order smoothness
    along the x-direction as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \bigg [ \frac{\partial m}{\partial x} \bigg ]^2 \, dv

    where :math:`m(r)` is the model and :math:`w(r)` is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \bigg | \, \frac{\partial m_i}{\partial x} \, \bigg |^2

    where :math:`m_i \in \mathbf{m}` are the discrete model parameter values defined on the mesh
    and :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that
    1) account for cell dimensions in the discretization and 2) apply any user-defined weighting.
    This is equivalent to an objective function of the form:

    .. math::
        \phi (\mathbf{m}) = \Big \| \mathbf{W \, G_x m } \, \Big \|^2

    where

        - :math:`\mathbf{G_x}` is the partial cell gradient operator along the x-direction, and
        - :math:`\mathbf{W}` is the weighting matrix.

    Note that since :math:`\mathbf{G_x}` maps from cell centers to x-faces,
    :math:`\mathbf{W}` is an operator that acts on variables living on x-faces.

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model :math:`\mathbf{m}^{(ref)}` can be
    preserved by including the reference model the regularization.
    In this case, the objective function becomes:

    .. math::
        \phi (\mathbf{m}) = \Big \| \mathbf{W G_x}
        \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    This functionality is used by setting a reference model with the
    `reference_model` property, and by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom weights defined on the faces specified by the `orientation` property; i.e. x-faces for
    smoothness along the x-direction. Each set of weights were either defined directly on the
    faces or have been averaged from cell centers.

    The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v_x} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v_x}` are cell volumes projected to x-faces.
    The weighting matrix is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom weights is stored within a ``dict`` as an ``numpy.ndarray``.
    A (n_cells, ) ``numpy.ndarray`` is used to define weights at cell centers, which are
    averaged to the appropriate faces internally when weighting is applied.
    A (n_faces, ) ``numpy.ndarray`` is used to define weights directly on the faces specified
    by the `orientation` input argument. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> array_1 = np.ones(mesh.n_cells)  # weights at cell centers
    >>> array_2 = np.ones(mesh.n_faces_x)  # weights directly on x-faces
    >>> reg = SmoothnessFirstOrder(
    >>>     mesh, orientation='x', weights={'weights_1': array_1, 'weights_2': array_2}
    >>> )

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2)

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

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

        Returns the partial gradient operator which takes the derivative along the
        orientation where smoothness is being enforced. For smoothness along the
        x-direction, the resulting operator would map from cell centers to x-faces.

        Returns
        -------
        scipy.sparse.csr_matrix
            Partial cell gradient operator defined on the
            :py:class:`.regularization.RegularizationMesh`.
        """
        if getattr(self, "_cell_gradient", None) is None:
            self._cell_gradient = getattr(
                self.regularization_mesh, "cell_gradient_{}".format(self.orientation)
            )
        return self._cell_gradient

    @property
    def reference_model_in_smooth(self) -> bool:
        # Inherited from BaseRegularization class
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
        r"""Evaluate the regularization kernel function.

        For first-order smoothness regularization in the x-direction,
        the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        where :math:`\mathbf{G_x}` is the partial cell gradient operator along the x-direction
        (i.e. x-derivative), :math:`\mathbf{m}` are the discrete model parameters defined on the
        mesh and :math:`\mathbf{m}^{(ref)}` is the reference model (optional).
        Similarly for smoothness along y and z.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function.

        Notes
        -----
        The objective function for first-order smoothness regularization along the x-direction
        is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, :math:`\mathbf{G_x}` is the partial
        cell gradient operator along the x-direction (i.e. x-derivative), and :math:`\mathbf{W}` is
        the weighting matrix. Similar for smoothness along y and z.
        See the :class:`SmoothnessFirstOrder` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W \, f_m} \Big \|^2
        """
        dfm_dl = self.mapping * self._delta_m(m)

        if self.units is not None and self.units.lower() == "radian":
            return (
                utils.mat_utils.coterminal(self.cell_gradient.sign() @ dfm_dl)
                / self._cell_distances
            )
        return self.cell_gradient @ dfm_dl

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the regularization kernel function.

        For first-order smoothness regularization in the x-direction, the derivative of the
        regularization kernel function with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{G_x}

        where :math:`\mathbf{G_x}` is the partial cell gradient operator along x
        (i.e. the x-derivative).

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the regularization kernel function.

        Notes
        -----
        The objective function for first-order smoothness regularization along the x-direction
        is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, :math:`\mathbf{G_x}` is the partial
        cell gradient operator along the x-direction (i.e. x-derivative), and :math:`\mathbf{W}` is
        the weighting matrix. Similar for smoothness along y and z.
        See the :class:`SmoothnessFirstOrder` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W \, f_m} \Big \|^2

        The derivative with respect to the model is therefore:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{G_x}
        """
        return self.cell_gradient @ self.mapping.deriv(self._delta_m(m))

    @property
    def W(self) -> csr_matrix:
        r"""Weighting matrix.

        Returns the weighting matrix for the objective function. To see how the
        weighting matrix is constructed, see the *Notes* section for the
        :class:`SmoothnessFirstOrder` regularization class.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the objective function.
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
        {'x','y','z'}
            The direction along which smoothness is enforced.

        """
        return self._orientation


class SmoothnessSecondOrder(SmoothnessFirstOrder):
    r"""Second-order smoothness (flatness) least-squares regularization.

    ``SmoothnessSecondOrder`` regularization is used to ensure that values in the recovered
    model have small second-order spatial derivatives. When a reference model is included,
    the regularization preserves second-order smoothness within the reference model along
    the direction specified (x, y or z). Optionally, custom cell weights can be used
    to control the degree of smoothness being enforced throughout different regions
    the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh mesh
        The mesh on which the regularization is discretized.
    orientation : {'x', 'y', 'z'}
        The direction along which smoothness is enforced.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active cells in the inversion.
        If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model. To include the reference model in the regularization, the
        `reference_model_in_smooth` property must be set to ``True``.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness regularization.
    units : None, str
        Units for the model parameters. Some regularization classes behave differently
        depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to
        a (n_cells, ) numpy.ndarray that is defined on the
        :py:class:`regularization.RegularizationMesh`.

    Notes
    -----
    We define the regularization function (objective function) for second-order
    smoothness along the x-direction as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \bigg [ \frac{\partial^2 m}{\partial x^2} \bigg ]^2 \, dv

    where :math:`m(r)` is the model and :math:`w(r)` is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \bigg | \, \frac{\partial^2 m_i}{\partial x^2} \, \bigg |^2

    where :math:`m_i \in \mathbf{m}` are the discrete model parameter values defined on the
    mesh and :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that
    1) account for cell dimensions in the discretization and 2) apply any user-defined weighting.
    This is equivalent to an objective function of the form:

    .. math::
        \phi (\mathbf{m}) = \big \| \mathbf{W \, L_x \, m } \, \big \|^2

    where

        - :math:`\mathbf{L_x}` is a second-order derivative operator with respect to :math:`x`, and
        - :math:`\mathbf{W}` is the weighting matrix.

    **Reference model in smoothness:**

    Second-order smoothness within a discrete reference model :math:`\mathbf{m}^{(ref)}` can be
    preserved by including the reference model the smoothness regularization function.
    In this case, the objective function becomes:

    .. math::
        \phi (\mathbf{m}) = \Big \| \mathbf{W L_x}
        \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    This functionality is used by setting a reference model with the
    `reference_model` property, and by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. The weighting applied within the objective function
    is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v}` are the cell volumes.
    The weighting matrix used to apply the weights is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = SmoothnessSecondOrder(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    """

    def f_m(self, m):
        r"""Evaluate the regularization kernel function.

        For second-order smoothness regularization in the x-direction,
        the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{L_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        where where :math:`\mathbf{m}` are the discrete model parameters (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model (optional), :math:`\mathbf{L_x}`
        is the discrete second order x-derivative operator.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function.

        Notes
        -----
        The objective function for second-order smoothness regularization along the x-direction
        is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W L_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, :math:`\mathbf{L_x}` is the
        second-order x-derivative operator, and :math:`\mathbf{W}` is
        the weighting matrix. Similar for smoothness along y and z.
        See the :class:`SmoothnessSecondOrder` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{L_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W \, f_m} \Big \|^2
        """
        dfm_dl = self.mapping * self._delta_m(m)

        if self.units is not None and self.units.lower() == "radian":
            return self.cell_gradient.T @ (
                utils.mat_utils.coterminal(self.cell_gradient.sign() @ dfm_dl)
                / self.length_scales
            )

        dfm_dl2 = self.cell_gradient @ dfm_dl

        return self.cell_gradient.T @ dfm_dl2

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the regularization kernel function.

        For second-order smoothness regularization, the derivative of the
        regularization kernel function with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{L_x}

        where :math:`\mathbf{L_x}` is the second-order derivative operator with respect to x.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the regularization kernel function.

        Notes
        -----
        The objective function for second-order smoothness regularization along the x-direction
        is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W L_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

        where :math:`\mathbf{m}` are the discrete model parameters (model),
        :math:`\mathbf{m}^{(ref)}` is the reference model, :math:`\mathbf{L_x}` is the
        second-order x-derivative operator, and :math:`\mathbf{W}` is
        the weighting matrix. Similar for smoothness along y and z.
        See the :class:`SmoothnessSecondOrder` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{L_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W \, f_m} \Big \|^2

        The derivative of the regularization kernel function with respect to the model is:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{L_x}
        """
        return (
            self.cell_gradient.T
            @ self.cell_gradient
            @ self.mapping.deriv(self._delta_m(m))
        )

    @property
    def W(self) -> csr_matrix:
        r"""Weighting matrix.

        Returns the weighting matrix for the objective function. To see how the
        weighting matrix is constructed, see the *Notes* section for the
        :class:`SmoothnessSecondOrder` regularization class.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the objective function.
        """
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
    r"""Weighted least-squares regularization using smallness and smoothness.

    Apply regularization using a weighted sum of :class:`Smallness`, :class:`SmoothnessFirstOrder`,
    and/or :class:`SmoothnessSecondOrder` (optional) least-squares regularization functions.
    ``Smallness`` regularization is used to ensure that values in the recovered model,
    or differences between the recovered model and a reference model, are not overly
    large in magnitude. ``Smoothness`` regularization is used to ensure that values in the
    recovered model are smooth along specified directions. When a reference model
    is included in the smoothness regularization, the inversion preserves
    gradients/interfaces within the reference model. Custom weights can also be supplied
    to control the degree of smallness and smoothness being
    enforced throughout different regions the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active cells in the inversion.
        If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness terms.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, )
        numpy.ndarray that is defined on the :py:class:`~.regularization.RegularizationMesh`.
    alpha_s : float, optional
        Scaling constant for the smallness regularization term.
    alpha_x, alpha_y, alpha_z : float or None, optional
        Scaling constants for the first order smoothness along x, y and z, respectively.
        If set to ``None``, the scaling constant is set automatically according to the
        value of the `length_scale` parameter.
    alpha_xx, alpha_yy, alpha_zz : 0, float
        Scaling constants for the second order smoothness along x, y and z, respectively.
        If set to ``None``, the scaling constant is set automatically according to the
        value of the `length_scale` parameter.
    length_scale_x, length_scale_y, length_scale_z : float, optional
        First order smoothness length scales for the respective dimensions.

    Notes
    -----
    Weighted least-squares regularization can be defined by a weighted sum of
    :class:`Smallness`, :class:`SmoothnessFirstOrder` and :class:`SmoothnessSecondOrder`
    regularization functions. This corresponds to a model objective function
    :math:`\phi_m (m)` of the form:

    .. math::
        \phi_m (m) =& \alpha_s \int_\Omega \, w(r)
        \Big [ m(r) - m^{(ref)}(r) \Big ]^2 \, dv \\
        &+ \sum_{j=x,y,z} \alpha_j \int_\Omega \, w(r)
        \bigg [ \frac{\partial m}{\partial \xi_j} \bigg ]^2 \, dv \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \int_\Omega \, w(r)
        \bigg [ \frac{\partial^2 m}{\partial \xi_j^2} \bigg ]^2 \, dv
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model, and :math:`w(r)`
    is a user-defined weighting function. :math:`\xi_j` is the unit direction along :math:`j`.
    Parameters :math:`\alpha_s`, :math:`\alpha_j` and :math:`\alpha_{jj}` for :math:`j=x,y,z`
    are multiplier constants which weight the respective contributions of the smallness and
    smoothness terms towards the regularization.

    For implementation within SimPEG, the regularization functions and their variables
    must be discretized onto a `mesh`. For a continuous variable :math:`x(r)` whose
    discrete representation on the mesh is given by :math:`\mathbf{x}`, we approximate
    as follows:

    .. math::
        \int_\Omega w(r) \big [ x(r) \big ]^2 \, dv \approx \sum_i \tilde{w}_i \, | x_i |^2

    where :math:`\tilde{w}_i` are amalgamated weighting constants that account for cell dimensions
    in the discretization and apply user-defined weighting. Using the above approximation,
    the ``WeightedLeastSquares`` regularization can be expressed as a weighted sum of
    objective functions of the form:

    .. math::
        \phi_m (\mathbf{m}) =& \alpha_s
        \Big \| \mathbf{W_s} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j G_j \, m} \, \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \Big \| \mathbf{W_{jj} L_j \, m} \, \Big \|^2
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    where

        - :math:`\mathbf{m}` are the set of discrete model parameters (i.e. the model),
        - :math:`\mathbf{m}^{(ref)}` is the reference model,
        - :math:`\mathbf{G_x, \, G_y, \; G_z}` are partial cell gradients operators along x, y and z,
        - :math:`\mathbf{L_x, \, L_y, \; L_z}` are second-order derivative operators with respect to x, y and z,
        - :math:`\mathbf{W_s, \, W_x, \, W_y, \; W_z}` are weighting matrices.

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model :math:`\mathbf{m}^{(ref)}` can be
    preserved by including the reference model the smoothness regularization.
    In this case, the objective function becomes:

    .. math::
        \phi_m (\mathbf{m}) =& \alpha_s
        \Big \| \mathbf{W_s} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j G_j}
        \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2 \\
        &+ \sum_{j=x,y,z} \alpha_{jj} \Big \| \mathbf{W_{jj} L_j}
        \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2
        \;\;\;\;\;\;\;\; \big ( \textrm{optional} \big )

    This functionality is used by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **Alphas and length scales:**

    The :math:`\alpha` parameters scale the relative contributions of the smallness and smoothness
    terms in the model objective function. Each :math:`\alpha` parameter can be set directly as a
    appropriate property of the ``WeightedLeastSquares`` class; e.g. :math:`\alpha_x` is set
    using the `alpha_x` property. Note that unless the parameters are set manually, second-order
    smoothness is not included in the model objective function. That is, the `alpha_xx`, `alpha_yy`
    and `alpha_zz` parameters are set to 0 by default.

    The relative contributions of smallness and smoothness terms on the recovered model can also
    be set by leaving `alpha_s` as its default value of 1, and setting the smoothness scaling
    constants based on length scales. The model objective function has been formulated such that
    smallness and smoothness terms contribute equally when the length scales are equal; i.e. when
    properties `length_scale_x = length_scale_y = length_scale_z`. When the `length_scale_x`
    property is set, the `alpha_x` and `alpha_xx` properties are set internally as:

    >>> reg.alpha_x = (reg.length_scale_x * reg.regularization_mesh.base_length) ** 2.0

    and

    >>> reg.alpha_xx = (ref.length_scale_x * reg.regularization_mesh.base_length) ** 4.0

    Likewise for y and z.

    **Custom weights and weighting matrices:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of custom
    cell weights that are applied to all terms in the model objective function.
    The general form for the weights applied to smallness and second-order smoothness terms
    is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v} \odot \prod_j \mathbf{w_j}

    and weights applied to first-order smoothness terms are given by:

    .. math::
        \mathbf{\tilde{w}} = \big ( \mathbf{P \, v} \big ) \odot \prod_j \mathbf{P \, w_j}

    :math:`\mathbf{v}` are the cell volumes, and :math:`\mathbf{P}` represents the
    projection matrix from cell centers to the appropriate faces;
    i.e. where discrete first-order derivatives live.

    Weights for each term are used to construct their respective weighting matrices
    as follows:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = WeightedLeastSquares(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})
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

        # Raise errors on deprecated arguments: avoid old code that still uses
        # them to silently fail
        if (key := "indActive") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. "
                "Please use 'active_cells' instead."
            )

        if (key := "cell_weights") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. Please use 'weights' instead."
            )

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

        # Check if weights is a dictionary, raise error if it's not
        if weights is not None and not isinstance(weights, dict):
            raise TypeError(
                f"Invalid 'weights' of type '{type(weights)}'. "
                "It must be a dictionary with strings as keys and arrays as values."
            )

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

        super().__init__(objfcts=objfcts, unpack_on_add=False, **kwargs)

        for fun in objfcts:
            fun.parent = self

        if active_cells is not None:
            self.active_cells = active_cells

        self.mapping = mapping
        self.reference_model = reference_model
        self.reference_model_in_smooth = reference_model_in_smooth
        self.alpha_xx = alpha_xx
        self.alpha_yy = alpha_yy
        self.alpha_zz = alpha_zz
        if weights is not None:
            self.set_weights(**weights)

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights for all child regularization objects.

        Parameters
        ----------
        **weights : key, numpy.ndarray
            Each keyword argument is added to the weights used by all child regularization objects.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import WeightedLeastSquares
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
            The name of the weights being removed from all child regularization objects.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import WeightedLeastSquares
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
        raise AttributeError(
            "'cell_weights' has been removed. "
            "Please access weights using the `set_weights`, `get_weights`, and "
            "`remove_weights` methods."
        )

    @cell_weights.setter
    def cell_weights(self, value):
        raise AttributeError(
            "'cell_weights' has been removed. "
            "Please access weights using the `set_weights`, `get_weights`, and "
            "`remove_weights` methods."
        )

    @property
    def alpha_s(self):
        """Multiplier constant for the smallness term.

        Returns
        -------
        float
            Multiplier constant for the smallness term.
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
        """Multiplier constant for first-order smoothness along x.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along x.
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
        """Multiplier constant for first-order smoothness along y.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along y.
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
        """Multiplier constant for first-order smoothness along z.

        Returns
        -------
        float
            Multiplier constant for first-order smoothness along z.
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
        """Multiplier constant for second-order smoothness along x.

        Returns
        -------
        float
            Multiplier constant for second-order smoothness along x.
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
        """Multiplier constant for second-order smoothness along y.

        Returns
        -------
        float
            Multiplier constant for second-order smoothness along y.
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
        """Multiplier constant for second-order smoothness along z.

        Returns
        -------
        float
            Multiplier constant for second-order smoothness along z.
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
        r"""Multiplier constant for smoothness along x relative to base scale length.

        Where the :math:`\Delta h` defines the base length scale (i.e. minimum cell dimension),
        and  :math:`\alpha_x` defines the multiplier constant for first-order smoothness along x,
        the length-scale is given by:

        .. math::
            L_x = \bigg ( \frac{\alpha_x}{\Delta h} \bigg )^{1/2}

        Returns
        -------
        float
            Multiplier constant for smoothness along x relative to base scale length.
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
        r"""Multiplier constant for smoothness along z relative to base scale length.

        Where the :math:`\Delta h` defines the base length scale (i.e. minimum cell dimension),
        and  :math:`\alpha_y` defines the multiplier constant for first-order smoothness along y,
        the length-scale is given by:

        .. math::
            L_y = \bigg ( \frac{\alpha_y}{\Delta h} \bigg )^{1/2}

        Returns
        -------
        float
            Multiplier constant for smoothness along z relative to base scale length.
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
        r"""Multiplier constant for smoothness along z relative to base scale length.

        Where the :math:`\Delta h` defines the base length scale (i.e. minimum cell dimension),
        and  :math:`\alpha_z` defines the multiplier constant for first-order smoothness along z,
        the length-scale is given by:

        .. math::
            L_z = \bigg ( \frac{\alpha_z}{\Delta h} \bigg )^{1/2}

        Returns
        -------
        float
            Multiplier constant for smoothness along z relative to base scale length.
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
        """Whether to include the reference model in the smoothness objective functions.

        Returns
        -------
        bool
            Whether to include the reference model in the smoothness objective functions.
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
        r"""Multiplier constants for weighted sum of objective functions.

        For a model objective function :math:`\phi_m (\mathbf{m})` constructed using
        a weighted sum of objective functions :math:`\phi_i (\mathbf{m})`, i.e.:

        .. math::
            \phi_m (\mathbf{m}) = \sum_i \alpha_i \, \phi_i (\mathbf{m})

        the `multipliers` property returns the list of multiplier constants :math:`alpha_i`
        in order.

        Returns
        -------
        list of float
            Multiplier constants for weighted sum of objective functions.
        """
        return [getattr(self, objfct._multiplier_pair) for objfct in self.objfcts]

    @property
    def active_cells(self) -> np.ndarray:
        """Active cells defined on the regularization mesh.

        A boolean array defining the cells in the :py:class:`~.regularization.RegularizationMesh`
        that are active (i.e. updated) throughout the inversion. The values of inactive cells
        remain equal to their starting model values.

        Returns
        -------
        (n_cells, ) array of bool

        Notes
        -----
        If the property is set using a ``numpy.ndarray`` of ``int``, the setter interprets the
        array as representing the indices of the active cells. When called however, the quantity
        will have been internally converted to a boolean array.
        """
        return self.regularization_mesh.active_cells

    @active_cells.setter
    def active_cells(self, values: np.ndarray):
        self.regularization_mesh.active_cells = values
        active_cells = self.regularization_mesh.active_cells
        # notify the objective functions that the active_cells changed
        for objfct in self.objfcts:
            objfct.active_cells = active_cells

    indActive = deprecate_property(
        active_cells,
        "indActive",
        "active_cells",
        "0.19.0",
        error=True,
    )

    @property
    def reference_model(self) -> np.ndarray:
        """Reference model.

        Returns
        -------
        None, (n_param, ) numpy.ndarray
            Reference model. If ``None``, the reference model in the inversion is set to
            the starting model.
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
        error=True,
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

        Mesh on which the regularization is discretized. This is not the same as
        the mesh on which the simulation is defined.

        Returns
        -------
        discretize.base.RegularizationMesh
            Mesh on which the regularization is discretized.
        """
        return self._regularization_mesh

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping from the model to the regularization mesh.

        Returns
        -------
        simpeg.maps.BaseMap
            The mapping from the model parameters to the quantity defined on the
            :py:class:`~simpeg.regularization.RegularizationMesh`.
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
    """Base regularization class for joint inversion.

    The ``BaseSimilarityMeasure`` class defines properties and methods used
    by regularization classes for joint inversion. It is not directly used to
    constrain inversions.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh
        Mesh on which the regularization is discretized. This is not necessarily the same as
        the mesh on which the simulation is defined.
    wire_map : simpeg.maps.WireMap
        Wire map connecting physical properties defined on active cells of the
        :class:`RegularizationMesh` to the entire model.
    """

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, **kwargs)
        self.wire_map = wire_map

    @property
    def wire_map(self):
        """Mapping from model to physical properties defined on the regularization mesh.

        Returns
        -------
        simpeg.maps.WireMap
            Mapping from model to physical properties defined on the regularization mesh.
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
        """Not implemented for ``BaseSimilarityMeasure`` class."""
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self, model, v=None):
        """Not implemented for ``BaseSimilarityMeasure`` class."""
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
        """Not implemented for ``BaseSimilarityMeasure`` class."""
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )
