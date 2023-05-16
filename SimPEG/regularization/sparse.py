from __future__ import annotations

import numpy as np

from discretize.base import BaseMesh

from .base import (
    BaseRegularization,
    WeightedLeastSquares,
    RegularizationMesh,
    Smallness,
    SmoothnessFirstOrder,
)
from .. import utils
from ..utils import (
    validate_ndarray_with_shape,
    validate_float,
    validate_type,
    validate_string,
)


class BaseSparse(BaseRegularization):
    """Base class for sparse-norm regularization.

    The ``BaseSparse`` class defines properties and methods inherited by sparse-norm
    regularization classes. Sparse-norm regularization in SimPEG is implemented using
    an iteratively re-weighted least squares (IRLS) approach. The ``BaseSparse`` class
    however, is not directly used to define the regularization for the inverse problem.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the
        regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the
        reference model is equal to the starting model for the inversion.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, )
        numpy.ndarray that is defined on the :py:class:`~.regularization.RegularizationMesh`.
    norm : float
        The norm used in the regularization function. Must be between within the interval [0, 2].
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function.
        If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    
    """

    def __init__(self, mesh, norm=2.0, irls_scaled=True, irls_threshold=1e-8, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.norm = norm
        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def irls_scaled(self) -> bool:
        """Scale IRLS weights.

        When ``True``, the iteratively re-weighted least-squares (IRLS) weights are scaled at
        each update to generally preserve the magnitude of the regularization term.

        Returns
        -------
        bool
            Whether to scale IRLS weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        self._irls_scaled = validate_type("irls_scaled", value, bool, cast=False)

    @property
    def irls_threshold(self):
        r"""Constant added to the denominator of the IRLS weights for stability.

        At each IRLS iteration, the IRLS weights are updated.
        The weight at iteration :math:`k` corresponding the cell :math:`i` is given by:

        .. math::
            r_i (\mathbf{m}^{(k)}) = \Big ( ( m_i^{(k-1)})^2 + \epsilon^2 \Big )^{p/2 - 1}

        where the `irls_threshold` :math:`\epsilon` is a constant that stabilizes the expression.

        Returns
        -------
        float
            Constant added to the denominator of the IRLS weights for stability.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        self._irls_threshold = validate_float(
            "irls_threshold", value, min_val=0.0, inclusive_min=False
        )

    @property
    def norm(self):
        r"""Norm for the sparse regularization.

        Sparse-norm regularization functions within SimPEG take the form:

        .. math::
            \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \, \Big ( \mathbb{F} \big [ m(r) - m_{ref}(r) \big ] \Bigg )^p \, dv

        where :math:`m(r)` is the model, :math:`m_{ref}(r)` is a reference model, and :math:`w(r)`
        is a user-defined weighting function. :math:`\mathbb{F}` is a placeholder for a function that
        acts on the difference between :math:`m` and :math:`m_{ref}`; e.g. a differential operator.
        
        The parameter :math:`p \in [0,2]` defines the norm, where a smaller norm is used to a model
        with more sparse structures.

        Returns
        -------
        None, float, (n_cells, ) numpy.ndarray
            Norm for the sparse regularization. If ``None``, a 2-norm is used.
            A float within the interval [0,2] represents a constant norm applied for all cells.
            A ``numpy.ndarray`` object, where each entry is used to apply a different norm to each cell in the mesh.
        """
        return self._norm

    @norm.setter
    def norm(self, value: float | np.ndarray | None):
        if value is None:
            value = np.ones(self._weights_shapes[0]) * 2.0
        expected_shapes = self._weights_shapes
        if isinstance(expected_shapes, list):
            expected_shapes = expected_shapes[0]
        value = validate_ndarray_with_shape(
            "norm", value, shape=[expected_shapes, (1,)], dtype=float
        )
        if value.shape == (1,):
            value = np.full(expected_shapes[0], value)

        if np.any(value < 0) or np.any(value > 2):
            raise ValueError(
                "Value provided for 'norm' should be in the interval [0, 2]"
            )
        self._norm = value

    def get_lp_weights(self, f_m):
        """Compute and return standard IRLS weights.

        The IRLS weights are applied to ensure the discrete approximation of the sparse
        regularization function is evaluated using the specified norm. Since the weighting
        is model dependent, it must be recomputed at every IRLS iteration.

        Parameters
        ----------
        f_m : fcn
            The least-squares regularization kernel.
        """
        lp_scale = np.ones_like(f_m)
        if self.irls_scaled:
            # Scale on l2-norm gradient: f_m.max()
            l2_max = np.ones_like(f_m) * np.abs(f_m).max()
            # Compute theoretical maximum gradients for p < 1
            l2_max[self.norm < 1] = self.irls_threshold / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            lp_values = l2_max / (l2_max**2.0 + self.irls_threshold**2.0) ** (
                1.0 - self.norm / 2.0
            )
            lp_scale[lp_values != 0] = np.abs(f_m).max() / lp_values[lp_values != 0]

        return lp_scale / (f_m**2.0 + self.irls_threshold**2.0) ** (
            1.0 - self.norm / 2.0
        )


class SparseSmallness(BaseSparse, Smallness):
    r"""Sparse smallness (compactness) regularization.

    ``SparseSmallness`` is used to recover models comprised of compact structures.
    The level of compactness is controlled by the norm within the regularization
    function; with more compact structures being recovered when a smaller norm is used.
    Optionally, custom cell weights can be included to control the degree of compactness
    being enforced throughout different regions the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : .regularization.RegularizationMesh
        Mesh on which the regularization is discretized. Not the mesh used to
        define the simulation.
    norm : float
        The norm used in the regularization function. Must be within theinterval [0, 2].
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`.regularization.RegularizationMesh` cells
        that are active in the inversion. If ``None``, all cells are active.
    mapping : None, .maps.IdentityMap
        The mapping from the model parameters to the quantity defined in the regularization.
        If ``None``, the mapping is the :class:`.maps.IdentityMap`.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the starting
        is set as the reference model by default.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to
        a (n_cells, ) numpy.ndarray that is defined on the
        :py:class:`regularization.RegularizationMesh`.
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function.
        If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.

    Notes
    -----
    We define the regularization function for sparse smallness (compactness) as:

    .. math::
        \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \,
        \Big | m(r) - m^{(ref)}(r) \Big |^p \, dv

    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model and :math:`w(r)`
    is a user-defined weighting function.
    
    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function is given by:

    .. math::
        \gamma (\mathbf{m}) = \frac{1}{2} \sum_i
        \tilde{w}_i \, \Big | m_i - m_i^{(ref)} \Big |^p

    where :math:`m_i \in \mathbf{m}` are the discrete model parameters defined on the mesh and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that accounts
    for cell dimensions in the discretization and applies user-defined weighting.

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \gamma \big (\mathbf{m}^{(k)} \big )
        = \frac{1}{2} \sum_i \tilde{w}_i \, \Big | m_i^{(k)} - m_i^{(ref)} \Big |^p
        \approx \frac{1}{2} \sum_i \tilde{w}_i \, r_i^{(k)} \Big | m_i^{(k)} - m_i^{(ref)} \Big |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \bigg [ \Big ( m_i^{(k-1)} - m_i^{(ref)} \Big )^2 +
        \epsilon^2 \; \bigg ]^{p/2 - 1}

    and :math:`\epsilon` is a small constant added for stability. For the set of model parameters
    :math:`\mathbf{m}` defined at cell centers, the convex least-squares problem for IRLS
    iteration :math:`k` can be expressed as follows:

    .. math::
        \gamma \big ( \mathbf{m}^{(k)} \big ) \approx \frac{1}{2} \Big \| \,
        \mathbf{W \, R} \, \big [ \mathbf{m}^{(k)} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    where

        - :math:`\mathbf{m}^{(k)}` are the discrete model parameters at iteration :math:`k`,
        - :math:`\mathbf{m}^{(ref)}` is a reference model (optional, set with `reference_model`),
        - :math:`\mathbf{R}` is the IRLS re-weighting matrix, and
        - :math:`\mathbf{W}` is the weighting matrix.

    **IRLS weights and the re-weighting matrix:**

    The IRLS weights are model-dependent and must be computed at every IRLS iteration.
    For iteration :math:`k`, the IRLS weights :math:`\mathbf{r_s}` are updated internally
    using the previous model via:

    .. math::
        \mathbf{r_s} \big ( \mathbf{m}^{(k)} \big ) = \lambda_s \big ( \mathbf{m}^{(k-1)} \big )
        \bigg [ \Big ( \mathbf{m}^{(k-1)} - \mathbf{m}^{(ref)} \Big )^2 + \epsilon^2 \bigg ]^{p/2 - 1}

    where :math:`\epsilon` is a small constant added for stability of the algorithm
    (set using `irls_threshold`). :math:`\lambda_s (\mathbf{m})` is an optional scaling
    constant (``bool`` set with `irls_scaled`). The scaling constant ensures the sparse
    norm has roughly the same magnitude as the equivalent 2-norm for the same model.
    The weights are then used to construct the IRLS re-weighting matrix :math:`\mathbf{R}`,
    where

    .. math::
        \mathbf{R} = \textrm{diag} \Big ( \mathbf{r_s}^{\! 1/2} \Big )

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. The weighting applied within the discrete regularization function
    is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{\tilde{v}} \odot \prod_k \mathbf{w_k}

    where :math:`\mathbf{\tilde{v}}` are default weights that account for cell volumes
    and dimensions when the regularization function is discretized to the mesh.
    The weighting matrix used to apply the weights is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = SparseSmallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    """

    _multiplier_pair = "alpha_s"

    def update_weights(self, m):
        r"""Compute and update the IRLS weights.

        The weights used to construct the re-weighting matrix :math:`\mathbf{R}` for
        sparse-norm inversion are model-dependent and must be updated at every IRLS iteration.
        This method recomputes and stores the weights. For a comprehensive description,
        see the *Notes* section in the :class:`SparseSmallness` class documentation.

        Parameters
        ----------
        m : numpy.ndarray
            The model.
        """
        f_m = self.f_m(m)
        self.set_weights(irls=self.get_lp_weights(f_m))


class SparseSmoothness(BaseSparse, SmoothnessFirstOrder):
    r"""Sparse smoothness (blockiness) regularization.

    ``SparseSmoothness`` is used to recover models comprised of blocky structures.
    The level of blockiness is controlled by the choice in norm within the regularization function;
    with more blocky structures being recovered when a smaller norm is used.
    Optionally, custom cell weights can be included to control the degree of blockiness being
    enforced throughout different regions the model.
    
    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : .regularization.RegularizationMesh
        Mesh on which the regularization is discretized. Not the mesh used to
        define the simulation.
    orientation : {'x','y','z'}
        The direction along which sparse smoothness is applied.
    norm : float
        The norm used in the regularization function. Must be within theinterval [0, 2].
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, .maps.IdentityMap
        The mapping from the model parameters to the quantity defined in the regularization.
        If ``None``, the mapping is the :class:`.maps.IdentityMap`.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the starting
        is set as the reference model by default.
    reference_model_in_smooth : bool, optional
        Whether to include the reference model in the smoothness terms.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Custom weights for the least-squares function. Each ``key`` points to
        a ``numpy.ndarray`` that is defined on the :py:class:`regularization.RegularizationMesh`.
        A (n_cells, ) ``numpy.ndarray`` is used to define weights at cell centers, which are
        averaged to the appropriate faces internally when weighting is applied.
        A (n_faces, ) ``numpy.ndarray`` is used to define weights directly on the faces specified
        by the `orientation` input argument.
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function.
        If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.
    gradient_type : {"total", "component"}
        Gradient measure used in the IRLS re-weighting. Whether to re-weight using the total
        gradient or components of the gradient.

    Notes
    -----
    We define the regularization function for sparse smoothness (blockiness) along the x-direction
    as:

    .. math::
        \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \,
        \bigg | \frac{\partial m}{\partial x} \bigg |^p \, dv

    where :math:`m(r)` is the model and :math:`w(r)` is a user-defined weighting function.
    The parameter :math:`p \in [0,2]` defines the `norm`, where a smaller norm is used to
    recovery structures with sharper boundaries.
    
    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discrete approximation for the regularization
    function is given by:

    .. math::
        \gamma (\mathbf{m}) = \frac{1}{2} \sum_i
        \tilde{w}_i \, \Bigg | \, \frac{\partial m_i}{\partial x} \, \Bigg |^p

    where :math:`m_i \in \mathbf{m}` are discrete model parameter values defined on the mesh and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that account
    for cell dimensions in the discretization and apply user-defined weighting.

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \gamma \big (\mathbf{m}^{(k)} \big )
        = \frac{1}{2} \sum_i
        \tilde{w}_i \, \Bigg | \, \frac{\partial m_i^{(k)}}{\partial x} \Bigg |^p
        \approx \frac{1}{2} \sum_i \tilde{w}_i \, r_i^{(k)}
        \Bigg | \, \frac{\partial m_i^{(k)}}{\partial x} \Bigg |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \Bigg [ \Bigg ( \frac{\partial m_i^{(k-1)}}{\partial x} \Bigg )^2 +
        \epsilon^2 \; \Bigg ]^{p/2 - 1}

    and :math:`\epsilon` is a small constant added for stability. For the set of model parameters
    :math:`\mathbf{m}` defined at cell centers, the convex least-squares problem for IRLS
    iteration :math:`k` can be expressed as follows:

    .. math::
        \gamma \big ( \mathbf{m}^{(k)} \big ) \approx \frac{1}{2} \Big \| \,
        \mathbf{W \, R \, G_x} \, \mathbf{m}^{(k)} \Big \|^2

    where

        - :math:`\mathbf{m}^{(k)}` are the discrete model parameters at iteration :math:`k`,
        - :math:`\mathbf{G_x}` is the partial cell-gradient operator along x (x-derivative),
        - :math:`\mathbf{R}` is the IRLS re-weighting matrix, and
        - :math:`\mathbf{W}` is the weighting matrix.

    Note that since :math:`\mathbf{G_x}` maps from cell centers to x-faces, :math:`\mathbf{R}` and
    :math:`\mathbf{W}` are operators that act on variables living on x-faces.

    **IRLS weights and the re-weighting matrix:**

    For smoothness along the x-direction, the IRLS weights :math:`\mathbf{r_x}` at every iteration
    are recomputed internally using the previous iteration via:

    .. math::
        \mathbf{r_x} \big ( \mathbf{m}^{(k)} \big ) = \lambda_x \big ( \mathbf{m}^{(k-1)} \big )
        \bigg [ \Big ( \mathbf{G_x} \big [ \mathbf{m}^{(k-1)} - \mathbf{m}^{(ref)} \big ] \Big )^2
        + \epsilon^2 \bigg ]^{q_x/2 - 1}

    where :math:`\epsilon` is a small constant added for stability of the algorithm
    (set using `irls_threshold`) and :math:`\lambda_x (\mathbf{m})` is an optional scaling constant
    (``bool`` set with `irls_scaled`). The scaling constant ensures the sparse
    norm has roughly the same magnitude as the equivalent 2-norm for the same model.
    The weights are then used to construct the IRLS re-weighting matrix :math:`\mathbf{R}`, where

    .. math::
        \mathbf{R} = \textrm{diag} \Big ( \mathbf{r_x}^{\! 1/2} \Big )

    Likewise for smoothness along y or z.

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model :math:`m^{(ref)}` can be preserved by
    including the reference model the smoothness regularization function. For smoothness along
    the x-direction, the regularization function is given by:

    .. math::
        \gamma (m) = \frac{1}{2} \int_\Omega \, w(r) \,
        \bigg [ \frac{\partial}{\partial x} \Big ( m - m^{(ref)} \Big ) \bigg ]^p \, dv

    When discretized onto a mesh, the regularization function becomes:
    
    .. math::
        \gamma (\mathbf{m}) = \frac{1}{2} \Big \| \mathbf{W \, R \, G_x}
        \big [ \mathbf{m}^{(k)} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    And the IRLS weights are at iteration :math:`k` are given by:

    .. math::
        \mathbf{r_x} \big ( \mathbf{m}^{(k)} \big ) = \lambda \big ( \mathbf{m}^{(k-1)} \big )
        \Bigg [ \bigg ( \mathbf{G_x} \Big [ \mathbf{m}^{(k-1)} - \mathbf{m}^{(ref)} \Big ] \bigg )^2
        + \epsilon^2 \bigg ]^{p/2 - 1}

    This functionality is used by setting :math:`\mathbf{m}^{(ref)}` with the
    `reference_model` property, and by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom weights defined on the faces specified by the `orientation` property; i.e. x-faces for
    smoothness along the x-direction. Each set of weights were either defined directly on the
    faces or have been averaged from cell centers.

    The weighting applied within the discrete regularization function is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{\tilde{v}} \odot \prod_k \mathbf{w_k}

    where :math:`\mathbf{\tilde{v}}` are default weights that account for cell volumes
    and dimensions when the regularization function is discretized onto the mesh.
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
    >>> reg = SparseSmoothness(
    >>>     mesh, orientation='x', weights={'weights_1': array_1, 'weights_2': array_2}
    >>> )

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    """

    def __init__(self, mesh, orientation="x", gradient_type="total", **kwargs):
        if "gradientType" in kwargs:
            self.gradientType = kwargs.pop("gradientType")
        else:
            self.gradient_type = gradient_type
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)

    def update_weights(self, m):
        r"""Compute and update the IRLS weights.

        The weights used to construct the re-weighting matrix :math:`\mathbf{R}` for
        sparse-norm inversion are model-dependent and must be updated at every IRLS iteration.
        This method recomputes and stores the weights. For a comprehensive description,
        see the *Notes* section in the :class:`SparseSmoothness` class documentation.

        Parameters
        ----------
        m : numpy.ndarray
            The model.
        """
        if self.gradient_type == "total":
            delta_m = self.mapping * self._delta_m(m)
            f_m = np.zeros_like(delta_m)
            for ii, comp in enumerate("xyz"):
                if self.regularization_mesh.dim > ii:
                    dm = (
                        getattr(self.regularization_mesh, f"cell_gradient_{comp}")
                        * delta_m
                    )

                    if self.units is not None and self.units.lower() == "radian":
                        Ave = getattr(self.regularization_mesh, f"aveCC2F{comp}")
                        length_scales = Ave * (
                            self.regularization_mesh.Pac.T
                            * self.regularization_mesh.mesh.h_gridded[:, ii]
                        )
                        dm = (
                            utils.mat_utils.coterminal(dm * length_scales)
                            / length_scales
                        )

                    f_m += np.abs(
                        getattr(self.regularization_mesh, f"aveF{comp}2CC") * dm
                    )

            f_m = getattr(self.regularization_mesh, f"aveCC2F{self.orientation}") * f_m

        else:
            f_m = self.f_m(m)

        self.set_weights(irls=self.get_lp_weights(f_m))

    @property
    def gradient_type(self) -> str:
        """Gradient measure used in the IRLS re-weighting.

        Returns
        -------
        str in {"total", "components"}
            Whether to re-weight using the total gradient or components of the gradient.
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        self._gradient_type = validate_string(
            "gradient_type", value, ["total", "components"]
        )

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.19.0", error=False, future_warn=True
    )


class Sparse(WeightedLeastSquares):
    r"""Sparse norm weighted least squares regularization.

    Construct a regularization for recovering compact and/or blocky structures
    using a weighted sum of :class:`SparseSmallness` and :class:`SparseSmoothness`
    regularization functions. The level of compactness and blockiness is
    controlled by the norms within the respective regularization functions;
    with more sparse structures (compact and/or blocky) being recovered when smaller
    norms are used. Optionally, custom cell weights can be applied to control
    the degree of sparseness being enforced throughout different regions the model.

    See the *Notes* section below for a comprehensive description.

    Parameters
    ----------
    mesh : SimPEG.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, numpy.ndarray of bool
        Array of bool defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    mapping : None, SimPEG.maps.BaseMap
        The mapping from the model parameters to the quantity defined in the
        regularization. If ``None``, the mapping is the identity map.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model values used to constrain the inversion. If ``None``, the
        reference model is equal to the starting model for the inversion.
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
    length_scale_x, length_scale_y, length_scale_z : float, optional
        First order smoothness length scales for the respective dimensions.
    gradient_type : {"total", "component"}
        Gradient measure used in the IRLS re-weighting. Whether to re-weight using the
        total gradient or components of the gradient.
    norms : (dim+1, ) numpy.ndarray
        The respective norms used for the sparse smallness, x-smoothness, (y-smoothness
        and z-smoothness) regularization function. Must all be within the interval [0, 2].
        E.g. `np.r_[2, 1, 1, 1]` uses a 2-norm on the smallness term and a 1-norm on all
        smoothness terms.
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization
        function. If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.

    Notes
    -----
    The model objective function :math:`\phi_m (m)` defined by a weighted sum of
    :class:`SparseSmallness` and :class:`SparseSmoothness` regularization functions
    is given by:

    .. math::
        \phi_m (m) = \frac{\alpha_s}{2} \int_\Omega \, w(r)
        \Big [ m(r) - m^{(ref)}(r) \Big ]^{p_s} \, dv
        + \sum_{j=x,y,z} \frac{\alpha_j}{2} \int_\Omega \, w(r)
        \bigg [ \frac{\partial m}{\partial \xi_j} \bigg ]^{q_j} \, dv
        
    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model, and :math:`w(r)`
    is a user-defined weighting function. :math:`\xi_j` is the unit direction along :math:`j`.
    Constants :math:`\alpha_s` and :math:`\alpha_j` for :math:`j=x,y,z` weight the respective
    contributions of the smallness and smoothness regularization functions.
    :math:`p_s` is the norm for the smallness term and :math:`q_j` for :math:`j=x,y,z`
    are the norms for the smoothness terms.

    For implementation within SimPEG, regularization functions and their variables
    must be discretized onto a `mesh`. For a continuous variable :math:`x(r)` whose
    discrete representation on the mesh is given by :math:`\mathbf{x}`, we approximate
    as follows:

    .. math::
        \int_\Omega w(r) \big [ x(r) \big ]^p \, dv \approx \sum_i \tilde{w}_i \, | x_i |^p

    where :math:`\tilde{w}_i` are amalgamated weighting constants that account for cell dimensions
    in the discretization and apply user-defined weighting.

    It is impractical to work with sparse norms directly, as their derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate sparse norms by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \sum_i \tilde{w}_i \, \Big | x_i^{(k)} \Big |^p
        \approx \sum_i \tilde{w}_i \, r_i^{(k)} \Big | x_i^{(k)} \Big |^2

    where IRLS weight :math:`r_i` for iteration :math:`k` is obtained from the previous iteration via:

    .. math::
        r_i^{(k)} = \bigg [ \Big ( x_i^{(k-1)} \Big )^2 + \epsilon^2 \; \bigg ]^{p/2 - 1}

    and :math:`\epsilon` is a small constant added for stability. For the set of model parameters
    :math:`\mathbf{m}` defined at cell centers, the convex least-squares problem for IRLS
    iteration :math:`k` can be expressed as follows:

    .. math::
        \phi_m (\mathbf{m}) = \frac{\alpha_s}{2}
        \Big \| \mathbf{W_s R_s} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2
        + \sum_{j=x,y,z} \frac{\alpha_j}{2} \Big \| \mathbf{W_j R_j G_j m} ] \Big \|^2

    where

        - :math:`\mathbf{m}` are the set of discrete model parameters (i.e. the model),
        - :math:`\mathbf{m}^{(ref)}` is the reference model,
        - :math:`\mathbf{G_x, \, G_y, \; G_z}` are partial cell gradients operators along x, y and z,
        - :math:`\mathbf{W_s, \, W_x, \, W_y, \; W_z}` are weighting matrices, and
        - :math:`\mathbf{R_s, \, R_x, \, R_y, \; R_z}` are IRLS re-weighting matrices

    **Alphas and length scales:**

    The :math:`\alpha` parameters scale the relative contributions of the smallness and smoothness
    terms in the model objective function. Each :math:`\alpha` parameter can be set directly as an
    appropriate property of the ``Sparse`` class; e.g. :math:`\alpha_x` is set
    using the `alpha_x` property. Note that unless the parameters are set manually, second-order
    smoothness is not included in the model objective function. That is, the `alpha_xx`, `alpha_yy`
    and `alpha_zz` properties are set to 0 by default.

    The model objective function has been formulated such that each term is
    roughly the same size when the :math:`\alpha` parameters are equal; e.g. when
    :math:`\alpha_s = \alpha_x = \alpha_y = \alpha_z` = ... This is accomplished by applying
    a default weighting to each term which negates the effects of cell size/dimensions.

    Smoothness parameters can also be set using length scales. For example, by setting the
    `length_scale_x` property, the `alpha_x` and `alpha_xx` properties are set as:

    >>> reg.alpha_x = (reg.length_scale_x * reg.regularization_mesh.base_length) ** 2.0

    and

    >>> reg.alpha_xx = (ref.length_scale_x * reg.regularization_mesh.base_length) ** 4.0

    Likewise for y and z.

    **Custom weights and weighting matrices:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of custom cell
    weights that are applied to all regularization terms in the model objective function.
    The general form for the weights applied to the smallness term is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{\tilde{v}} \odot \prod_k \mathbf{w_k}

    And the general form for the weights applied to first-order smoothness terms is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{\tilde{v}} \odot \prod_k \mathbf{P \, w_k}

    :math:`\mathbf{\tilde{v}}` is a placeholder for the default weights that account for cell volumes
    and dimensions when regularization functions are discretized onto the mesh. These default
    weights vary depending on the regularization term; e.g. smallness, first-order smoothness in
    the y-direction. For first-order smoothness terms, a projection matrix :math:`\mathbf{P}` is
    required to average weights at cell centers to the appropriate faces; i.e. where discrete
    first-order derivatives live.

    Weights for each regularization term are used to construct their respective weighting matrices
    as follows:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = Sparse(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    **IRLS weights and the re-weighting matrix:**

    IRLS weights allow each term in the sparse regularization to be approximated by a weighted 2-norm.
    The IRLS weights for each term are model-dependent and must be computed at every IRLS iteration.
    For iteration :math:`k`, the IRLS weights :math:`\mathbf{r_s}` for the smallness term
    are updated internally using the previous model via:

    .. math::
        \mathbf{r_s} \big ( \mathbf{m}^{(k)} \big ) =  \lambda_s \big ( \mathbf{m}^{(k-1)} \big )
        \bigg [ \Big ( \big ( \mathbf{m}^{(k-1)} \big ) \Big )^2 + \epsilon^2 \bigg ]^{p_s/2 - 1}

    For first-order smoothness along the x-direction (likewise for smoothness in y and z),
    the IRLS weights :math:`\mathbf{r_x}` are updated via:

    .. math::
        \mathbf{r_x} \big ( \mathbf{m}^{(k)} \big ) =  \lambda_x \big ( \mathbf{m}^{(k-1)} \big )
        \bigg [ \Big ( \big ( \mathbf{G_x m}^{(k-1)} \big ) \Big )^2 + \epsilon^2 \bigg ]^{q_x/2 - 1}

    where :math:`\epsilon` is a small constant added for stability of the algorithm
    (set using `irls_threshold`). :math:`\lambda` are optional scaling constants
    (``bool`` set with `irls_scaled`) applied to each set of IRLS weights.
    The scaling constant ensures the sparse norm has roughly the same magnitude as the
    equivalent 2-norm for the same model. The IRLS weights are then used to construct their
    respective re-weighting matrices :math:`\mathbf{R_\ast}` as follows:

    .. math::
        \mathbf{R_\ast} = \textrm{diag} \Big ( \mathbf{r_\ast}^{\! 1/2} \Big )

    where :math:`\ast` represents :math:`s, x, y` or :math:`z`.
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        norms=None,
        gradient_type="total",
        irls_scaled=True,
        irls_threshold=1e-8,
        **kwargs,
    ):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh
        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        objfcts = [
            SparseSmallness(mesh=self.regularization_mesh),
            SparseSmoothness(mesh=self.regularization_mesh, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(
                SparseSmoothness(mesh=self.regularization_mesh, orientation="y")
            )

        if mesh.dim > 2:
            objfcts.append(
                SparseSmoothness(mesh=self.regularization_mesh, orientation="z")
            )

        gradientType = kwargs.pop("gradientType", None)
        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )
        if norms is None:
            norms = [1] * (mesh.dim + 1)
        self.norms = norms

        if gradientType is not None:
            # Trigger deprecation warning
            self.gradientType = gradientType
        else:
            self.gradient_type = gradient_type

        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def gradient_type(self) -> str:
        """Choice of gradient measure for IRLS weights.

        Returns
        -------
        {"total", "component"}
            Choice of gradient measure for IRLS weights.
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        for fct in self.objfcts:
            if hasattr(fct, "gradient_type"):
                fct.gradient_type = value

        self._gradient_type = value

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.19.0", error=False, future_warn=True
    )

    @property
    def norms(self):
        """Norms for the child regularization classes.

        Supply the norms for the smallness and all smoothness terms in the ``Sparse`` regularization.
        
        Returns
        -------
        list of float or numpy.ndarray
            Norms for the child regularization classes.
        """
        return self._norms

    @norms.setter
    def norms(self, values: list | np.ndarray | None):
        if values is not None:
            if len(values) != len(self.objfcts):
                raise ValueError(
                    f"The number of values provided for 'norms', {len(values)}, does not "
                    f"match the number of regularization functions, {len(self.objfcts)}."
                )
        else:
            values = [None] * len(self.objfcts)
        previous_norms = getattr(self, "_norms", [None] * len(self.objfcts))
        try:
            for val, fct in zip(values, self.objfcts):
                fct.norm = val
            self._norms = values
        except Exception as err:
            # reset the norms if failed
            for val, fct in zip(previous_norms, self.objfcts):
                fct.norm = val
            raise err

    @property
    def irls_scaled(self) -> bool:
        """Scale IRLS weights.

        Returns
        -------
        bool
            Scale the IRLS weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        value = validate_type("irls_scaled", value, bool, cast=False)
        for fct in self.objfcts:
            fct.irls_scaled = value
        self._irls_scaled = value

    @property
    def irls_threshold(self):
        """IRLS stabilization constant.

        Constant added to the denominator of the IRLS weights for stability.
        See documentation for the :class:`Sparse` class for a comprehensive description.

        Returns
        -------
        float
            IRLS stabilization constant.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        value = validate_float(
            "irls_threshold", value, min_val=0.0, inclusive_min=False
        )
        for fct in self.objfcts:
            fct.irls_threshold = value
        self._irls_threshold = value

    def update_weights(self, model):
        """Trigger IRLS update for all child regularization functions.

        Parameters
        ----------
        model
            The model.
        """
        for fct in self.objfcts:
            fct.update_weights(model)
