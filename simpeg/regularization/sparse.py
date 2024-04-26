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
        Reference model values used to constrain the inversion. If ``None``, the starting model
        is set as the reference model.
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

        When ``True``, scaling is applied when computing IRLS weights.
        The scaling acts to preserve the balance between the data misfit and the components of
        the regularization based on the derivative of the l2-norm measure. And it assists the
        convergence by ensuring the model does not deviate
        aggressively from the global 2-norm solution during the first few IRLS iterations.
        For a comprehensive description, see the documentation for :py:meth:`get_lp_weights` .

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
        r"""Stability constant for computing IRLS weights.

        Returns
        -------
        float
            Stability constant for computing IRLS weights.
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
        r"""Compute and return iteratively re-weighted least-squares (IRLS) weights.

        For a regularization kernel function :math:`\mathbf{f_m}(\mathbf{m})`
        evaluated at model :math:`\mathbf{m}`, compute and return the IRLS weights.
        See :py:meth:`Smallness.f_m` and :py:meth:`SmoothnessFirstOrder.f_m` for examples of
        least-squares regularization kernels.

        For :class:`SparseSmallness`, *f_m* is a (n_cells, ) ``numpy.ndarray``.
        For :class:`SparseSmoothness`, *f_m* is a ``numpy.ndarray`` whose length corresponds
        to the number of faces along a particular orientation; e.g. for smoothness along x,
        the length is (n_faces_x, ).

        Parameters
        ----------
        f_m : numpy.ndarray
            The regularization kernel function evaluated at the current model.

        Notes
        -----
        For a regularization kernel function :math:`\mathbf{f_m}` evaluated at model
        :math:`\mathbf{m}`, the IRLS weights are computed via:

        .. math::
            \mathbf{w_r} = \boldsymbol{\lambda} \oslash
            \Big [ \mathbf{f_m}^{\!\! 2} + \epsilon^2 \Big ]^{1 - \mathbf{p}/2}

        where :math:`\oslash` represents elementwise division, :math:`\epsilon` is a small
        constant added for stability of the algorithm (set using the `irls_threshold` property),
        and :math:`\mathbf{p}` defines the `norm` at each cell.

        :math:`\boldsymbol{\lambda}` applies optional scaling to the IRLS weights
        (when the `irls_scaled` property is ``True``).
        The scaling acts to preserve the balance between the data misfit and the components of
        the regularization based on the derivative of the l2-norm measure. And it assists the
        convergence by ensuring the model does not deviate
        aggressively from the global 2-norm solution during the first few IRLS iterations.

        To apply elementwise scaling, let

        .. math::
            f_{max} = \big \| \, \mathbf{f_m} \, \big \|_\infty

        And define a vector array :math:`\mathbf{\tilde{f}_{\! max}}` such that:

        .. math::
            \tilde{f}_{\! i,max} = \begin{cases}
            f_{max} \;\;\; for \; p_i \geq 1 \\
            \frac{\epsilon}{\sqrt{1 - p_i}} \;\;\;\;\;\;\, for \; p_i < 1
            \end{cases}

        The elementwise scaling vector :math:`\boldsymbol{\lambda}` is:

        .. math::
            \boldsymbol{\lambda} = \bigg [ \frac{f_{max}}{\mathbf{\tilde{f}_{max}}} \bigg ]
            \odot \bigg [ \mathbf{f_{\! max}}^{\!\! 2} + \epsilon^2} \bigg ]^{1 - \mathbf{p}/2}
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
    norm : float, (n_cells, ) array_like
        The norm defining sparseness in the regularization function. Use a ``float`` to define
        the same norm for all mesh cells, or define an independent norm for each cell. All norm
        values must be within the interval [0, 2].
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
        :py:class:`regularization.RegularizationMesh`.
    irls_scaled : bool
        If ``True``, scale the IRLS weights to preserve magnitude of the regularization function.
        If ``False``, do not scale.
    irls_threshold : float
        Constant added to IRLS weights to ensures stability in the algorithm.

    Notes
    -----
    We define the regularization function (objective function) for sparse smallness (compactness) as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \Big | \, m(r) - m^{(ref)}(r) \, \Big |^{p(r)} \, dv

    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model, :math:`w(r)`
    is a user-defined weighting function and :math:`p(r) \in [0,2]` is a parameter which imposes
    sparseness throughout the recovered model. More compact structures are recovered in regions
    where :math:`p` is small. If the same level of sparseness is being imposed everywhere,
    the exponent becomes a constant.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \Big | m_i - m_i^{(ref)} \Big |^{p_i}

    where :math:`m_i \in \mathbf{m}` are the discrete model parameters defined on the mesh.
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply user-defined weighting.
    :math:`p_i \in \mathbf{p}` define the norm for each cell (set using `norm`).

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \phi \big (\mathbf{m}^{(k)} \big )
        = \sum_i \tilde{w}_i \, \Big | m_i^{(k)} - m_i^{(ref)} \Big |^{p_i}
        \approx \sum_i \tilde{w}_i \, r_i^{(k)} \Big | m_i^{(k)} - m_i^{(ref)} \Big |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \bigg [ \Big ( m_i^{(k-1)} - m_i^{(ref)} \Big )^2 +
        \epsilon^2 \; \bigg ]^{{p_i}/2 - 1}

    and :math:`\epsilon` is a small constant added for stability (set using `irls_threshold`).
    For the set of model parameters :math:`\mathbf{m}` defined at cell centers, the objective
    function for IRLS iteration :math:`k` can be expressed as follows:

    .. math::
        \phi \big ( \mathbf{m}^{(k)} \big ) \approx \Big \| \,
        \mathbf{W}^{\! (k)} \big [ \mathbf{m}^{(k)} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    where

        - :math:`\mathbf{m}^{(k)}` are the discrete model parameters at iteration :math:`k`,
        - :math:`\mathbf{m}^{(ref)}` is a reference model (optional, set with `reference_model`),
        - :math:`\mathbf{W}^{(k)}` is the weighting matrix for iteration :math:`k`. It applies the IRLS weights, user-defined weighting, and accounts for cell dimensions when the regularization function is discretized.

    **IRLS weights, user-defined weighting and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. And let :math:`\mathbf{r_s}^{\!\! (k)}` represent the IRLS weights
    for iteration :math:`k`. The net weighting applied within the objective function
    is given by:

    .. math::
        \mathbf{w}^{(k)} = \mathbf{r_s}^{\!\! (k)} \odot \mathbf{v} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v}` are the cell volumes.
    For a description of how IRLS weights are updated at every iteration, see the documentation
    for :py:meth:`update_weights`.

    The weighting matrix used to apply the weights is given by:

    .. math::
        \mathbf{W}^{(k)} = \textrm{diag} \Big ( \sqrt{\mathbf{w}^{(k)} \, } \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = SparseSmallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    """

    _multiplier_pair = "alpha_s"

    def update_weights(self, m):
        r"""Update the IRLS weights for sparse smallness regularization.

        Parameters
        ----------
        m : numpy.ndarray
            The model used to update the IRLS weights.

        Notes
        -----
        For the model :math:`\mathbf{m}` provided, the regularization kernel function
        for sparse smallness is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{m} - \mathbf{m}^{(ref)}

        where :math:`\mathbf{m}^{(ref)}` is the reference model; see :py:meth:`Smallness.f_m`
        for a more comprehensive definition.

        The IRLS weights are computed via:

        .. math::
            \mathbf{w_r} = \boldsymbol{\lambda} \oslash
            \Big [ \mathbf{f_m}^{\!\! 2} + \epsilon^2 \Big ]^{1 - \mathbf{p}/2}

        where :math:`\oslash` represents elementwise division, :math:`\epsilon` is a small
        constant added for stability of the algorithm (set using the `irls_threshold` property),
        and :math:`\mathbf{p}` defines the norm for each cell (defined using the `norm` property).

        :math:`\boldsymbol{\lambda}` applies optional scaling to the IRLS weights
        (when the `irls_scaled` property is ``True``).
        The scaling acts to preserve the balance between the data misfit and the components of
        the regularization based on the derivative of the l2-norm measure. And it assists the
        convergence by ensuring the model does not deviate
        aggressively from the global 2-norm solution during the first few IRLS iterations.

        To compute the scaling, let

        .. math::
            f_{max} = \big \| \, \mathbf{f_m} \, \big \|_\infty

        and define a vector array :math:`\mathbf{\tilde{f}_{\! max}}` such that:

        .. math::
            \tilde{f}_{\! i,max} = \begin{cases}
            f_{max} \;\;\;\;\; for \; p_i \geq 1 \\
            \frac{\epsilon}{\sqrt{1 - p_i}} \;\;\; for \; p_i < 1
            \end{cases}

        The scaling quantity :math:`\boldsymbol{\lambda}` is:

        .. math::
            \boldsymbol{\lambda} = \Bigg [ \frac{f_{max}}{\mathbf{\tilde{f}_{max}}} \Bigg ]
            \odot \Big [ \mathbf{\tilde{f}_{max}}^{\!\! 2} + \epsilon^2 \Big ]^{1 - \mathbf{p}/2}
        """
        f_m = self.f_m(m)
        self.set_weights(irls=self.get_lp_weights(f_m))


class SparseSmoothness(BaseSparse, SmoothnessFirstOrder):
    r"""Sparse smoothness (blockiness) regularization.

    ``SparseSmoothness`` is used to recover models comprised of blocky structures.
    The level of blockiness is controlled by the choice in norm within the regularization
    function; with more blocky structures being recovered when a smaller norm is used.
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
    norm : float, array_like
        The norm defining sparseness thoughout the regularization function. Must be within the
        interval [0,2]. There are several options:

        - ``float``: constant sparse norm throughout the domain.
        - (n_faces, ) ``array_like``: define the sparse norm independently at each face set by `orientation` (e.g. x-faces).
        - (n_cells, ) ``array_like``: define the sparse norm independently for each cell. Will be averaged to faces specified by `orientation` (e.g. x-faces).

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
    The regularization function (objective function) for sparse smoothness (blockiness)
    along the x-direction as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \Bigg | \, \frac{\partial m}{\partial x} \, \Bigg |^{p(r)} \, dv

    where :math:`m(r)` is the model, :math:`w(r)`
    is a user-defined weighting function and :math:`p(r) \in [0,2]` is a parameter which imposes
    sparseness throughout the recovered model. Sharper boundaries are recovered in regions
    where :math:`p(r)` is small. If the same level of sparseness is being imposed everywhere,
    the exponent becomes a constant.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discrete approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \Bigg | \, \frac{\partial m_i}{\partial x} \, \Bigg |^{p_i}

    where :math:`m_i \in \mathbf{m}` are the discrete model parameters defined on the mesh.
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply user-defined weighting.
    :math:`p_i \in \mathbf{p}` define the norm for each face (set using `norm`).

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \phi \big (\mathbf{m}^{(k)} \big )
        = \sum_i
        \tilde{w}_i \, \Bigg | \, \frac{\partial m_i^{(k)}}{\partial x} \Bigg |^{p_i}
        \approx \sum_i \tilde{w}_i \, r_i^{(k)}
        \Bigg | \, \frac{\partial m_i^{(k)}}{\partial x} \Bigg |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \Bigg [ \Bigg ( \frac{\partial m_i^{(k-1)}}{\partial x} \Bigg )^2 +
        \epsilon^2 \; \Bigg ]^{{p_i}/2 - 1}

    and :math:`\epsilon` is a small constant added for stability (set using `irls_threshold`).
    For the set of model parameters :math:`\mathbf{m}` defined at cell centers, the objective
    function for IRLS iteration :math:`k` can be expressed as follows:

    .. math::
        \phi \big ( \mathbf{m}^{(k)} \big ) \approx \Big \| \,
        \mathbf{W}^{(k)} \, \mathbf{G_x} \, \mathbf{m}^{(k)} \Big \|^2

    where

        - :math:`\mathbf{m}^{(k)}` are the discrete model parameters at iteration :math:`k`,
        - :math:`\mathbf{G_x}` is the partial cell-gradient operator along x (x-derivative),
        - :math:`\mathbf{W}^{(k)}` is the weighting matrix for iteration :math:`k`. It applies the IRLS weights, user-defined weighting, and accounts for cell dimensions when the regularization function is discretized.

    Note that since :math:`\mathbf{G_x}` maps from cell centers to x-faces, the weighting matrix
    acts on variables living on x-faces.

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model :math:`\mathbf{m}^{(ref)}` can be
    preserved by including the reference model the smoothness regularization.
    In this case, the least-squares problem for IRLS iteration :math:`k` becomes:

    .. math::
        \phi \big ( \mathbf{m}^{(k)} \big ) \approx \Big \| \,
        \mathbf{W}^{(k)} \mathbf{G_x}
        \big [ \mathbf{m}^{(k)} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    This functionality is used by setting :math:`\mathbf{m}^{(ref)}` with the
    `reference_model` property, and by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **IRLS weights, user-defined weighting and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom weights defined on the faces specified by the `orientation` property;
    i.e. x-faces for smoothness along the x-direction. Each set of weights were either defined
    directly on the faces or have been averaged from cell centers. And let
    :math:`\mathbf{r_x}^{\!\! (k)}` represent the IRLS weights for iteration :math:`k`.
    The net weighting applied within the objective function is given by:

    .. math::
        \mathbf{w}^{(k)} = \mathbf{r_x}^{\!\! (k)} \odot \mathbf{v_x} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v_x}` are cell volumes projected to x-faces; i.e. where the
    x-derivative lives. For a description of how IRLS weights are updated at every iteration,
    see the documentation for :py:meth:`update_weights`.

    The weighting matrix used to apply the weights is given by:

    .. math::
        \mathbf{W}^{(k)} = \textrm{diag} \Big ( \sqrt{\mathbf{w}^{(k)} \, } \Big )

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
        # Raise error if removed arguments were passed
        if (key := "gradientType") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. "
                "Please use 'gradient_type' instead."
            )
        self.gradient_type = gradient_type
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)

    def update_weights(self, m):
        r"""Update the IRLS weights for sparse smoothness regularization.

        Parameters
        ----------
        m : numpy.ndarray
            The model used to update the IRLS weights.

        Notes
        -----
        Let us consider the IRLS weights for sparse smoothness along the x-direction.
        When the class property `gradient_type`=`'components'`, IRLS weights are computed
        using the regularization kernel function and we define:

        .. math::
            \mathbf{f_m} = \mathbf{G_x} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ]

        where :math:`\mathbf{m}` is the model provided, :math:`\mathbf{G_x}` is the partial cell
        gradient operator along x (i.e. x-derivative), and :math:`\mathbf{m}^{(ref)}` is a
        reference model (optional, activated using `reference_model_in_smooth`).
        See :py:meth:`SmoothnessFirstOrder.f_m` for a more comprehensive definition of the
        regularization kernel function.

        However, when the class property `gradient_type`=`'total'`, IRLS weights are computed
        using the magnitude of the total gradient and we define:

        .. math::
            \mathbf{{f}_m} = \mathbf{A_{cx}} \sum_{j=x,y,z} \Big | \mathbf{A_j G_j}
            \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big |

        where :math:`\mathbf{A_j}` for :math:`j=x,y,z` averages the partial gradients from their
        respective faces to cell centers, and :math:`\mathbf{A_{cx}}` averages the sum of the
        absolute values back to the appropriate faces.

        Once :math:`\mathbf{f_m}` is obtained, the IRLS weights are computed via:

        .. math::
            \mathbf{w_r} = \boldsymbol{\lambda} \oslash
            \Big [ \mathbf{f_m}^{\!\! 2} + \epsilon^2 \Big ]^{1 - \mathbf{p}/2}

        where :math:`\oslash` represents elementwise division, :math:`\epsilon` is a small
        constant added for stability of the algorithm (set using the `irls_threshold` property),
        and :math:`\mathbf{p}` defines the norm for each element (set using the `norm` property).

        :math:`\boldsymbol{\lambda}` applies optional scaling to the IRLS weights
        (when the `irls_scaled` property is ``True``).
        The scaling acts to preserve the balance between the data misfit and the components of
        the regularization based on the derivative of the l2-norm measure. And it assists the
        convergence by ensuring the model does not deviate
        aggressively from the global 2-norm solution during the first few IRLS iterations.

        To apply the scaling, let

        .. math::
            f_{max} = \big \| \, \mathbf{f_m} \, \big \|_\infty

        and define a vector array :math:`\mathbf{\tilde{f}_{\! max}}` such that:

        .. math::
            \tilde{f}_{\! i,max} = \begin{cases}
            f_{max} \;\;\;\;\; for \; p_i \geq 1 \\
            \frac{\epsilon}{\sqrt{1 - p_i}} \;\;\; for \; p_i < 1
            \end{cases}

        The scaling vector :math:`\boldsymbol{\lambda}` is:

        .. math::
            \boldsymbol{\lambda} = \Bigg [ \frac{f_{max}}{\mathbf{\tilde{f}_{max}}} \Bigg ]
            \odot \Big [ \mathbf{\tilde{f}_{max}}^{\!\! 2} + \epsilon^2 \Big ]^{1 - \mathbf{p}/2}
        """
        if self.gradient_type == "total" and self.parent is not None:
            f_m = np.zeros(self.regularization_mesh.nC)
            for obj in self.parent.objfcts:
                if isinstance(obj, SparseSmoothness):
                    avg = getattr(self.regularization_mesh, f"aveF{obj.orientation}2CC")
                    f_m += np.abs(avg * obj.f_m(m))

            f_m = getattr(self.regularization_mesh, f"aveCC2F{self.orientation}") * f_m

        else:
            f_m = self.f_m(m)

        self.set_weights(irls=self.get_lp_weights(f_m))

    @property
    def gradient_type(self) -> str:
        """Gradient measure used to update IRLS weights for sparse smoothness.

        This property specifies whether the IRLS weights for sparse smoothness regularization
        are updated using the total gradient (*"total"*) or using the partial gradient along
        the smoothing orientation (*"components"*). To see how the IRLS weights are computed,
        visit the documentation for :py:meth:`update_weights`.

        Returns
        -------
        str in {"total", "components"}
            Whether to re-weight using the total gradient or partial gradients along
            smoothing orientations.
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        self._gradient_type = validate_string(
            "gradient_type", value, ["total", "components"]
        )

    gradientType = utils.code_utils.deprecate_property(
        gradient_type,
        "gradientType",
        new_name="gradient_type",
        removal_version="0.19.0",
        error=True,
    )


class Sparse(WeightedLeastSquares):
    r"""Sparse norm weighted least squares regularization.

    Apply regularization for recovering compact and/or blocky structures
    using a weighted sum of :class:`SparseSmallness` and :class:`SparseSmoothness`
    regularization functions. The level of compactness and blockiness is
    controlled by the norms within the respective regularization functions;
    with more sparse structures (compact and/or blocky) being recovered when smaller
    norms are used. Optionally, custom cell weights can be applied to control
    the degree of sparseness being enforced throughout different regions the model.

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
    Sparse regularization can be defined by a weighted sum of
    :class:`SparseSmallness`  and :class:`SparseSmoothness`
    regularization functions. This corresponds to a model objective function
    :math:`\phi_m (m)` of the form:

    .. math::
        \phi_m (m) = \alpha_s \int_\Omega \, w(r)
        \Big | \, m(r) - m^{(ref)}(r) \, \Big |^{p_s(r)} \, dv
        + \sum_{j=x,y,z} \alpha_j \int_\Omega \, w(r)
        \Bigg | \, \frac{\partial m}{\partial \xi_j} \, \Bigg |^{p_j(r)} \, dv

    where :math:`m(r)` is the model, :math:`m^{(ref)}(r)` is the reference model, and :math:`w(r)`
    is a user-defined weighting function applied to all terms.
    :math:`\xi_j` for :math:`j=x,y,z` are unit directions along :math:`j`.
    Parameters :math:`\alpha_s` and :math:`\alpha_j` for :math:`j=x,y,z` are multiplier constants
    that weight the respective contributions of the smallness and smoothness terms in the
    regularization. :math:`p_s(r) \in [0,2]` is a parameter which imposes sparse smallness
    throughout the recovered model; where more compact structures are recovered in regions where
    :math:`p_s(r)` is small. And :math:`p_j(r) \in [0,2]` for :math:`j=x,y,z` are parameters which
    impose sparse smoothness throughout the recovered model along the specified direction;
    where sharper boundaries are recovered in regions where these parameters are small.

    For implementation within SimPEG, regularization functions and their variables
    must be discretized onto a `mesh`. For a regularization function whose kernel is given by
    :math:`f(r)`, we approximate as follows:

    .. math::
        \int_\Omega w(r) \big [ f(r) \big ]^{p(r)} \, dv \approx \sum_i \tilde{w}_i \, | f_i |^{p_i}

    where :math:`f_i \in \mathbf{f_m}` define the discrete regularization kernel function
    on the mesh. For example, the regularization kernel function for smallness regularization
    is:

    .. math::
        \mathbf{f_m}(\mathbf{m}) = \mathbf{m - m}^{(ref)}

    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply user-defined weighting.
    :math:`p_i \in \mathbf{p}` define the sparseness throughout the domain (set using `norm`).

    It is impractical to work with sparse norms directly, as their derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate sparse norms by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \sum_i \tilde{w}_i \, \Big | f_i^{(k)} \Big |^{p_i}
        \approx \sum_i \tilde{w}_i \, r_i^{(k)} \Big | f_i^{(k)} \Big |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \bigg [ \Big ( f_i^{(k-1)} \Big )^2 + \epsilon^2 \; \bigg ]^{p_i/2 - 1}

    and :math:`\epsilon` is a small constant added for stability (set using `irls_threshold`).
    For the set of model parameters :math:`\mathbf{m}` defined at cell centers, the model
    objective function for IRLS iteration :math:`k` can be expressed as a weighted sum of
    objective functions of the form:

    .. math::
        \phi_m (\mathbf{m}) = \alpha_s
        \Big \| \mathbf{W_s}^{\!\! (k)} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2
        + \sum_{j=x,y,z} \alpha_j \Big \| \mathbf{W_j}^{\! (k)} \mathbf{G_j \, m} \Big \|^2

    where

        - :math:`\mathbf{m}` are the set of discrete model parameters (i.e. the model),
        - :math:`\mathbf{m}^{(ref)}` is the reference model,
        - :math:`\mathbf{G_x, \, G_y, \; G_z}` are partial cell gradients operators along x, y and z, and
        - :math:`\mathbf{W_s, \, W_x, \, W_y, \; W_z}` are the weighting matrices for iteration :math:`k`.

    The weighting matrices apply the IRLS weights, user-defined weighting, and account for cell
    dimensions when the regularization functions are discretized.

    **IRLS weights, user-defined weighting and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of custom cell
    weights that are applied to all objective functions in the model objective function.
    For IRLS iteration :math:`k`, the general form for the weights applied to the sparse smallness
    term is given by:

    .. math::
        \mathbf{w_s}^{\!\! (k)} = \mathbf{r_s}^{\!\! (k)} \odot
        \mathbf{v} \odot \prod_j \mathbf{w_j}

    And for sparse smoothness along x (likewise for y and z) is given by:

    .. math::
        \mathbf{w_x}^{\!\! (k)} = \mathbf{r_x}^{\!\! (k)} \odot \big ( \mathbf{P_x \, v} \big )
        \odot \prod_j \mathbf{P_x \, w_j}

    The IRLS weights at iteration :math:`k` are defined as :math:`\mathbf{r_\ast}^{\!\! (k)}`
    for :math:`\ast = s,x,y,z`. :math:`\mathbf{v}` are the cell volumes.
    Operators :math:`\mathbf{P_\ast}` for :math:`\ast = x,y,z`
    project to the appropriate faces.

    Once the net weights for all objective functions are computed,
    their weighting matrices can be constructed via:

    .. math::
        \mathbf{W}_\ast^{(k)} = \textrm{diag} \Big ( \, \sqrt{\mathbf{w_\ast}^{\!\! (k)} \, } \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = Sparse(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model can be preserved by including the
    reference model the smoothness regularization. In this case,
    the objective function becomes:

    .. math::
        \phi_m (\mathbf{m}) = \alpha_s
        \Big \| \mathbf{W_s}^{\! (k)} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2
        + \sum_{j=x,y,z} \alpha_j \Big \|
        \mathbf{W_j}^{\! (k)} \mathbf{G_j} \big [ \mathbf{m} - \mathbf{m}^{(ref)} \big ] \Big \|^2

    This functionality is used by setting the `reference_model_in_smooth` parameter
    to ``True``.

    **Alphas and length scales:**

    The :math:`\alpha` parameters scale the relative contributions of the smallness and smoothness
    terms in the model objective function. Each :math:`\alpha` parameter can be set directly as an
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
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        norms=None,
        gradient_type="total",
        irls_scaled=True,
        irls_threshold=1e-8,
        objfcts=None,
        **kwargs,
    ):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )

        # Raise error if removed arguments were passed
        if (key := "gradientType") in kwargs:
            raise TypeError(
                f"'{key}' argument has been removed. "
                "Please use 'gradient_type' instead."
            )

        self._regularization_mesh = mesh
        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        if objfcts is None:
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

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )
        if norms is None:
            norms = [1] * (mesh.dim + 1)
        self.norms = norms
        self.gradient_type = gradient_type
        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def gradient_type(self) -> str:
        """Gradient measure used to update IRLS weights for sparse smoothness.

        This property specifies whether the IRLS weights for sparse smoothness regularization(s)
        terms are updated using the total gradient (*"total"*) or using the partial gradients along
        their smoothing orientations (*"components"*). To see how the IRLS weights are computed,
        visit the documentation for :py:meth:`~SparseSmoothness.update_weights`.

        Returns
        -------
        str in {"total", "components"}
            Whether to re-weight using the total gradient or partial gradients along
            smoothing orientations.
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        for fct in self.objfcts:
            if hasattr(fct, "gradient_type"):
                fct.gradient_type = value

        self._gradient_type = value

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.19.0", error=True
    )

    @property
    def norms(self):
        """Norms for the child regularization classes.

        Norms for the smallness and all smoothness terms in the ``Sparse`` regularization.

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
        """Update IRLS weights for all child regularization objects.

        For an instance of the `Sparse` regularization class, this method re-computes and updates
        the IRLS for all child regularization objects using the model provided.
        To see how IRLS weights are recomputed for :class:`SparseSmallness` objects, visit the
        documentation for :py:meth:`SparseSmallness.update_weights`. And for
        :class:`SparseSmoothness` objects, visit the documentation for
        :py:meth:`SparseSmoothness.update_weights`.

        Parameters
        ----------
        model : (n_params, ) numpy.ndarray
            The model used to recompute the IRLS weights.
        """
        for fct in self.objfcts:
            fct.update_weights(model)
