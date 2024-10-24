import numpy as np
from discretize.base import BaseMesh
from .. import maps
from ..objective_function import BaseObjectiveFunction, ComboObjectiveFunction
from .. import utils
from .regularization_mesh import FaceRegularizationMesh

from simpeg.utils.code_utils import deprecate_property, validate_ndarray_with_shape

from scipy.sparse import csr_matrix


class BaseFaceRegularization(BaseObjectiveFunction):
    """Base regularization class.

    The ``BaseFaceRegularization`` class defines properties and methods inherited by
    SimPEG regularization classes, and is not directly used to construct inversions.

    Parameters
    ----------
    mesh : simpeg.regularization.FaceRegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_faces : None, (n_faces, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.FaceRegularizationMesh`
        faces that are active in the inversion. If ``None``, all faces are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active faces in the inversion.
        If ``None``, the mapping is set to :obj:`simpeg.maps.IdentityMap`.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function.
        Each value is a numpy.ndarray of shape(:py:property:`~.regularization.FaceRegularizationMesh.n_faces`, ).

    """

    _model = None
    _parent = None
    _W = None

    def __init__(
        self,
        mesh: FaceRegularizationMesh | BaseMesh,
        active_faces: np.ndarray | None = None,
        mapping: maps.IdentityMap | None = None,
        reference_model: np.ndarray | None = None,
        units: str | None = None,
        weights: dict | None = None,
        **kwargs,
    ):
        if isinstance(mesh, BaseMesh):
            mesh = FaceRegularizationMesh(mesh)

        if not isinstance(mesh, FaceRegularizationMesh):
            raise TypeError(
                f"'regularization_mesh' must be of type {FaceRegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        if weights is not None and not isinstance(weights, dict):
            raise TypeError(
                f"Invalid 'weights' of type '{type(weights)}'. "
                "It must be a dictionary with strings as keys and arrays as values."
            )

        super().__init__(nP=None, mapping=None, **kwargs)
        self._regularization_mesh = mesh
        self._weights = {}
        if active_faces is not None:
            self.active_faces = active_faces
        self.mapping = mapping  # Set mapping using the setter
        self.reference_model = reference_model
        self.units = units
        if weights is not None:
            self.set_weights(**weights)

    @property
    def active_faces(self) -> np.ndarray:
        """Active faces defined on the regularization mesh.

        A boolean array defining the faces in the :py:class:`~.regularization.FaceRegularizationMesh`
        that are active (i.e. updated) throughout the inversion. The values of inactive faces
        remain equal to their starting model values.

        Returns
        -------
        (n_faces, ) array of bool

        Notes
        -----
        If the property is set using a ``numpy.ndarray`` of ``int``, the setter interprets the
        array as representing the indices of the active faces. When called however, the quantity
        will have been internally converted to a boolean array.
        """
        return self.regularization_mesh.active_faces

    @active_faces.setter
    def active_faces(self, values: np.ndarray | None):
        self.regularization_mesh.active_faces = values

        if values is not None:
            area_term = "area" in self._weights
            self._weights = {}
            self._W = None
            if area_term:
                self.set_weights(area=self.regularization_mesh.active_areas)

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
            :py:class:`~.regularization.FaceRegularizationMesh`.
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
            and self.regularization_mesh.nF != "*"
        ):
            return (self.regularization_mesh.nF,)

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
    def regularization_mesh(self) -> FaceRegularizationMesh:
        """Face regularization mesh.

        Mesh on which the regularization is discretized. This is not the same as
        the mesh on which the simulation is defined. See :class:`.regularization.FaceRegularizationMesh`

        Returns
        -------
        .regularization.FaceRegularizationMesh
            Mesh on which the regularization is discretized.
        """
        return self._regularization_mesh

    def get_weights(self, key) -> np.ndarray:
        """Face weights for a given key.

        Parameters
        ----------
        key: str
            Name of the weights requested.

        Returns
        -------
        (n_faces, ) numpy.ndarray
            Face weights for a given key.

        Examples
        --------
        >>> import discretize
        >>> from simpeg.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_faces))
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
        >>> reg.set_weights(my_weight=np.ones(mesh.n_faces))
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
        >>> reg.set_weights(my_weight=np.ones(mesh.n_faces))
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

        nF = getattr(self.regularization_mesh, "nF", None)
        mapping = getattr(self, "_mapping", None)

        if mapping is not None and mapping.shape[1] != "*":
            return self.mapping.shape[1]

        if nF != "*" and nF is not None:
            return self.regularization_mesh.n_faces

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
        """Not implemented for ``BaseFaceRegularization`` class."""
        raise AttributeError("Regularization class must have a 'f_m' implementation.")

    def f_m_deriv(self, m) -> csr_matrix:
        """Not implemented for ``BaseFaceRegularization`` class."""
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


class FaceSmallness(BaseFaceRegularization):
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
    mesh : .regularization.FaceRegularizationMesh
        Mesh on which the regularization is discretized. Not the mesh used to
        define the simulation.
    active_faces : None, (n_faces, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.FaceRegularizationMesh`
        faces that are active in the inversion. If ``None``, all faces are active.
    mapping : None, simpeg.maps.BaseMap
        The mapping from the model parameters to the active faces in the inversion.
        If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to
        a (n_faces, ) numpy.ndarray that is defined on the
        :py:class:`regularization.FaceRegularizationMesh` .

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

    where :math:`\mathbf{v}` are the cell areas.
    The weighting matrix used to apply the weights for smallness regularization is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_faces, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = Smallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2)

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('area')

    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        active_areas = self.regularization_mesh.active_areas
        self.set_weights(area=active_areas)

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

