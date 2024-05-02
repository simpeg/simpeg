from __future__ import annotations
from typing import TYPE_CHECKING

import scipy.sparse as sp
import numpy as np
from .base import Smallness
from discretize.base import BaseMesh
from .base import RegularizationMesh, BaseRegularization
from .sparse import Sparse, SparseSmallness, SparseSmoothness

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseVectorRegularization(BaseRegularization):
    """Base regularization class for models defined by vector quantities.

    The ``BaseVectorRegularization`` class defines properties and methods used
    by regularization classes for inversion to recover vector quantities.
    It is not directly used to constrain inversions.
    """

    @property
    def n_comp(self):
        """Number of components in the model."""
        if self.mapping.shape[0] == "*":
            return self.regularization_mesh.dim
        return int(self.mapping.shape[0] / self.regularization_mesh.nC)

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        mesh = self.regularization_mesh

        return [(mesh.nC,), (self.n_comp * mesh.nC,), (mesh.nC, self.n_comp)]


class CrossReferenceRegularization(Smallness, BaseVectorRegularization):
    r"""Cross reference regularization for models representing vector quantities.

    ``CrossReferenceRegularization`` encourages the vectors in the recovered model to
    be oriented in the same directions as the vector in a reference vector model.
    The regularization function (objective function) constrains the inversion by penalizing
    the magnitude of the-cross product of the vector model with a reference vector model.
    The cross product, and therefore the objective function, is minimized when vectors
    in the model and reference vector model are parallel (or anti-parallel) to each other.
    And it is maximized when the vectors are perpendicular to each other.
    The reference vector model can be set using a single vector, or by defining a
    vector for each mesh cell.

    Parameters
    ----------
    mesh : discretize.base.BaseMesh, .RegularizationMesh
        The mesh defining the model discretization.
    ref_dir : (mesh.dim,) array_like or (mesh.dim, n_active) array_like
        The reference direction model. This can be either a constant vector applied
        to every model cell, or different for every active model cell.
    active_cells : index_array, optional
        Boolean array or an array of active indices indicating the active cells of the
        inversion domain mesh.
    mapping : simpeg.maps.IdentityMap, optional
        An optional linear mapping that would go from the model space to the space where
        the cross-product is enforced.
    weights : dict of [str: array_like], optional
        Any cell based weights for the regularization. Note if given a weight that is
        (n_cells, dim), meaning it is dependent on the vector component, it will compute
        the geometric mean of the component weights per cell and use that as a weight.
    **kwargs
        Arguments passed on to the parent classes: :py:class:`.Smallness` and
        :py:class:`.BaseVectorRegularization`.

    Notes
    -----
    Consider the case where the model is a vector quantity :math:`\vec{m}(r)`.
    The regularization function (objective function) for cross-reference
    regularization is given by:

    .. math::
        \phi (\vec{m}) = \int_\Omega \, \vec{w}(r) \, \cdot \,
        \Big [ \vec{m}(r) \, \times \, \vec{m}^{(ref)}(r) \Big ]^2 \, dv

    where :math:`\vec{m}^{(ref)}(r)` is the reference model vector and :math:`\vec{w}(r)`
    is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is given by:

    .. math::
        \phi (\vec{m}) \approx \sum_i \tilde{w}_i \, \cdot \,
        \Big | \vec{m}_i \, \times \, \vec{m}_i^{(ref)} \Big |^2

    where :math:`\tilde{m}_i \in \mathbf{m}` are the model vectors at cell centers and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply any user-defined weighting.

    In practice, the model is a discrete vector of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m_p} \\ \mathbf{m_s} \\ \mathbf{m_t} \end{bmatrix}

    where :math:`\mathbf{m_p}`, :math:`\mathbf{m_s}` and :math:`\mathbf{m_t}` represent vector
    components in the primary, secondary and tertiary directions at cell centers, respectively.
    The cross product between :math:`\mathbf{m}` and a similar reference vector
    :math:`\mathbf{m^{(ref)}}` (set with `ref_dir`) is a linear operation of the form:

    .. math::
        \mathbf{m} \times \mathbf{m^{ref}} = \mathbf{X m} =
        \begin{bmatrix}
        \mathbf{0} & -\boldsymbol{\Lambda_s} & \boldsymbol{\Lambda_t} \\
        \boldsymbol{\Lambda_p} & \mathbf{0} & -\boldsymbol{\Lambda_t} \\
        -\boldsymbol{\Lambda_p} & \boldsymbol{\Lambda_s} & \mathbf{0}
        \end{bmatrix} \!
        \begin{bmatrix} \mathbf{m_p} \\ \mathbf{m_s} \\ \mathbf{m_t} \end{bmatrix}

    where :math:`\mathbf{X}` is a linear operator that applies the cross-product on :math:`\mathbf{m}`,
    :math:`\mathbf{W}` is the weighting matrix, and:

    .. math::
        \boldsymbol{\Lambda_j} = \textrm{diag} \Big ( \mathbf{m_j^{(ref)}} \Big )
        \;\;\;\; \textrm{for} \; j=p,s,t

    The discrete regularization function in linear form can ultimately be expressed as:

    .. math::
        \phi (\mathbf{m}) =
        \Big \| \mathbf{W X m} \, \Big \|^2


    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. Each individual set of cell weights acts independently on the
    the directional components of the cross-product and is a discrete vector of the form:

    .. math::
        \mathbf{w_j} = \begin{bmatrix} \mathbf{w_p} \\ \mathbf{w_s} \\ \mathbf{w_t} \end{bmatrix}

    The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \big ( \mathbf{e_3 \otimes v} ) \odot \prod_j \mathbf{w_j}

    where

        - :math:`\mathbf{e_3}` is a vector of ones of length 3,
        - :math:`\otimes` is the Kronecker product, and
        - :math:`\mathbf{v}` are the cell volumes.

    The weighting matrix used to apply the weights in the regularization function is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as a ``list`` of (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = CrossReferenceRegularization(
    >>>     mesh, weights={'weights_1': array_1, 'weights_2': array_2}
    >>> )

    where `array_1` and `array_2` are (n_cells, dim) ``numpy.ndarray``.
    Weights can also be set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

    """

    def __init__(
        self, mesh, ref_dir, active_cells=None, mapping=None, weights=None, **kwargs
    ):
        kwargs.pop("reference_model", None)
        super().__init__(
            mesh=mesh,
            active_cells=active_cells,
            mapping=mapping,
            weights=weights,
            **kwargs,
        )
        self.ref_dir = ref_dir
        self.reference_model = 0.0

    @property
    def _nC_residual(self):
        return np.prod(self.ref_dir.shape)

    @property
    def ref_dir(self):
        """The reference direction model.

        Returns
        -------
        (n_active, dim) numpy.ndarray
            The reference direction model.
        """
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, value):
        mesh = self.regularization_mesh
        nC = mesh.nC
        value = np.asarray(value)
        if value.shape != (nC, mesh.dim):
            if value.shape == (mesh.dim,):
                # expand it out for each mesh cell
                value = np.tile(value, (nC, 1))
            else:
                raise ValueError(f"ref_dir must be shape {(nC, mesh.dim)}")
        self._ref_dir = value

        R0 = sp.diags(value[:, 0])
        R1 = sp.diags(value[:, 1])
        if value.shape[1] == 2:
            X = sp.bmat([[R1, -R0]])
        elif value.shape[1] == 3:
            Z = sp.csr_matrix((nC, nC))
            R2 = sp.diags(value[:, 2])
            X = sp.bmat(
                [
                    [Z, R2, -R1],
                    [-R2, Z, R0],
                    [R1, -R0, Z],
                ]
            )
        self._X = X

    def f_m(self, m):
        r"""Evaluate the regularization kernel function.

        For cross reference regularization, the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        where :math:`\mathbf{m}` are the discrete model parameters and :math:`\mathbf{X}`
        carries out the cross-product with a reference vector model.
        For a more detailed description, see the *Notes* section below.

        Parameters
        ----------
        m : numpy.ndarray
            The vector model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function evaluated for the model provided.

        Notes
        -----
        The objective function for cross reference regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W X m} \, \Big \|^2

        where :math:`\mathbf{m}` are the discrete vector model parameters defined on the mesh (model),
        :math:`\mathbf{X}` carries out the cross-product with a reference vector model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`CrossReferenceRegularization` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        """
        return self._X @ (self.mapping * m)

    def f_m_deriv(self, m):
        r"""Derivative of the regularization kernel function.

        For ``CrossReferenceRegularization``, the derivative of the regularization kernel function
        with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{X}

        where :math:`\mathbf{X}` is a linear operator that carries out the
        cross-product with a reference vector model.

        Parameters
        ----------
        m : numpy.ndarray
            The vector model.

        Returns
        -------
        scipy.sparse.csr_matrix
            The derivative of the regularization kernel function.

        Notes
        -----
        The objective function for cross reference regularization is given by:

        .. math::
            \phi_m (\mathbf{m}) =
            \Big \| \mathbf{W X m} \, \Big \|^2

        where :math:`\mathbf{m}` are the discrete vector model parameters defined on the mesh (model),
        :math:`\mathbf{X}` carries out the cross-product with a reference vector model, and :math:`\mathbf{W}` is
        the weighting matrix. See the :class:`CrossReferenceRegularization` class documentation for more detail.

        We define the regularization kernel function :math:`\mathbf{f_m}` as:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{X m}

        such that

        .. math::
            \phi_m (\mathbf{m}) = \Big \| \mathbf{W} \, \mathbf{f_m} \Big \|^2

        Thus, the derivative with respect to the model is:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} = \mathbf{X}

        """
        return self._X @ self.mapping.deriv(m)

    @property
    def W(self):
        r"""Weighting matrix.

        Returns the weighting matrix for the objective function. To see how the
        weighting matrix is constructed, see the *Notes* section for the
        :class:`CrossReferenceRegularization` class.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the objective function.
        """
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh
            nC = mesh.nC

            weights = np.ones(
                nC,
            )
            for value in self._weights.values():
                if value.shape == (nC,):
                    weights *= value
                elif value.size == mesh.dim * nC:
                    weights *= np.linalg.norm(
                        value.reshape((nC, mesh.dim), order="F"), axis=1
                    )
            weights = np.sqrt(weights)
            if mesh.dim == 2:
                diag = weights
            else:
                diag = np.r_[weights, weights, weights]
            self._W = sp.diags(diag, format="csr")
        return self._W


class BaseAmplitude(BaseVectorRegularization):
    """Base amplitude regularization class for models defined by vector quantities.

    The ``BaseAmplitude`` class defines properties and methods used
    by amplitude regularization classes for vector quantities.
    It is not directly used to constrain inversions.
    """

    def amplitude(self, m):
        """Return vector amplitudes for the model provided.

        Where the model `m` defines a vector quantity for each active cell in the
        inversion, the `amplitude` method returns the amplitudes of these vectors.

        Parameters
        ----------
        m : (n_param ) numpy.ndarray
            The model.

        Returns
        -------
        (n_cells, ) numpy.ndarray
            The amplitudes of the vectors for the model provided.
        """
        return np.linalg.norm(
            (self.mapping * self._delta_m(m)).reshape(
                (self.regularization_mesh.nC, self.n_comp), order="F"
            ),
            axis=1,
        )

    def deriv(self, m) -> np.ndarray:
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters;
        i.e. the gradient:

        .. math::
            \frac{\partial \phi}{\partial \mathbf{m}}

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model for which the gradient is evaluated.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Gradient of the regularization function evaluated for the model provided.
        """
        d_m = self._delta_m(m)

        return (
            2
            * self.f_m_deriv(m).T
            * (
                self.W.T
                @ self.W
                @ (self.f_m_deriv(m) @ d_m).reshape((-1, self.n_comp), order="F")
            ).flatten(order="F")
        )

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
            return (
                2
                * f_m_deriv.T
                * (sp.block_diag([self.W.T * self.W] * self.n_comp) * f_m_deriv)
            )

        return (
            2
            * f_m_deriv.T
            * (
                self.W.T
                @ self.W
                @ (f_m_deriv * v).reshape((-1, self.n_comp), order="F")
            ).flatten(order="F")
        )


class AmplitudeSmallness(SparseSmallness, BaseAmplitude):
    r"""Sparse smallness regularization on vector amplitudes.

    ``AmplitudeSmallness`` is a sparse norm smallness regularization that acts on the
    amplitudes of the vectors defining the model. Sparse norm functionality allows the
    use to recover more compact regions of higher amplitude vectors.
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
    We define the regularization function (objective function) for sparse amplitude smallness
    (compactness) as:

    .. math::
        \phi (\vec{m}) = \int_\Omega \, w(r) \,
        \Big | \, \vec{m}(r) - \vec{m}^{(ref)}(r) \, \Big |^{p(r)} \, dv

    where :math:`\vec{m}(r)` is the model, :math:`\vec{m}^{(ref)}(r)` is the reference model, :math:`w(r)`
    is a user-defined weighting function and :math:`p(r) \in [0,2]` is a parameter which imposes
    sparseness throughout the recovered model. More compact structures are recovered in regions
    where :math:`p` is small. If the same level of sparseness is being imposed everywhere,
    the exponent becomes a constant.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \Big | \vec{m}_i - \vec{m}_i^{(ref)} \Big |^{p_i}

    where :math:`\mathbf{m}` are the model parameters, :math:`\vec{m}_i` represents the vector
    defined for mesh cell :math:`i`, and :math:`\vec{m}_i^{(ref)}` defines the reference model
    vector for cell :math:`i`. :math:`\tilde{w}_i` are amalgamated weighting constants that
    1) account for cell dimensions in the discretization and 2) apply user-defined weighting.
    :math:`p_i \in \mathbf{p}` define the norm for each cell (set using `norm`).

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \phi \big (\mathbf{m}^{(k)} \big )
        = \sum_i \tilde{w}_i \, \Big | \, \vec{m}_i^{(k)} - \vec{m}_i^{(ref)} \, \Big |^{p_i}
        \approx \sum_i \tilde{w}_i \, r_i^{(k)}
        \Big | \, \vec{m}_i^{(k)} - \vec{m}_i^{(ref)} \, \Big |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \bigg [ \, \Big | \vec{m}_i^{(k-1)} - \vec{m}_i^{(ref)} \Big |^2 +
        \epsilon^2 \; \bigg ]^{{p_i}/2 - 1}

    and :math:`\epsilon` is a small constant added for stability (set using `irls_threshold`).

    The global set of model parameters :math:`\mathbf{m}` defined at cell centers is ordered according
    to its primary (:math:`p`), secondary (:math:`s`) and tertiary (:math:`t`) directions as follows:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m}_p \\ \mathbf{m}_s \\ \mathbf{m}_t \end{bmatrix}

    We define the amplitudes of the residual between the model and reference model for all cells as:

    .. math::
        \mathbf{\bar{m}} = \bigg (
        \Big [ \mathbf{m}_p - \mathbf{m}_p^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_s - \mathbf{m}_s^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_t - \mathbf{m}_t^{(ref)} \Big ]^2 \bigg )^{1/2}

    The objective function for IRLS iteration :math:`k` is given by:

    .. math::
        \phi \big ( \mathbf{\bar{m}}^{(k)} \big ) \approx \Big \| \,
        \mathbf{W}^{(k)} \, \mathbf{\bar{m}}^{(k)} \; \Big \|^2

    where

        - :math:`\mathbf{\bar{m}}^{(k)}` are the absolute values of the residual at iteration :math:`k`, and
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

    >>> reg = AmplitudeSmallness(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})
    """

    def f_m(self, m):
        r"""Evaluate the regularization kernel function.

        For smallness vector amplitude regularization, the regularization kernel function is:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{\bar{m}} = \bigg (
            \Big [ \mathbf{m}_p - \mathbf{m}_p^{(ref)} \Big ]^2 +
            \Big [ \mathbf{m}_s - \mathbf{m}_s^{(ref)} \Big ]^2 +
            \Big [ \mathbf{m}_t - \mathbf{m}_t^{(ref)} \Big ]^2 \bigg )^{1/2}

        where the global set of model parameters :math:`\mathbf{m}` defined at cell centers is
        ordered according to its primary (:math:`p`), secondary (:math:`s`) and tertiary (:math:`t`)
        directions as follows:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m}_p \\ \mathbf{m}_s \\ \mathbf{m}_t \end{bmatrix}

        Likewise for the vector components of the reference model.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function evaluated at the model provided.

        """

        return self.amplitude(m)

    @property
    def W(self):
        ### Inherited from Smallness regularization class
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh
            nC = mesh.nC

            weights = np.ones(
                nC,
            )
            for value in self._weights.values():
                if value.shape == (nC,):
                    weights *= value
                elif value.shape == (self.n_comp * nC,):
                    weights *= np.linalg.norm(
                        value.reshape((nC, self.n_comp), order="F"), axis=1
                    )

            self._W = sp.diags(np.sqrt(weights), format="csr")

        return self._W


class AmplitudeSmoothnessFirstOrder(SparseSmoothness, BaseAmplitude):
    r"""Sparse amplitude smoothness (blockiness) regularization.

    ``AmplitudeSmallness`` is a sparse norm smoothness regularization that acts on the
    amplitudes of the vectors defining the model. Sparse norm functionality allows the
    use to recover more blocky regions of higher amplitude vectors.
    The level of blockiness is controlled by the norm within the regularization
    function; with more blocky structures being recovered when a smaller norm is used.
    Optionally, custom cell weights can be included to control the degree of blockiness
    being enforced throughout different regions the model.

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
    The regularization function (objective function) for sparse amplitude smoothness (blockiness)
    along the x-direction as:

    .. math::
        \phi (m) = \int_\Omega \, w(r) \,
        \Bigg | \, \frac{\partial |\vec{m}|}{\partial x} \, \Bigg |^{p(r)} \, dv

    where :math:`\vec{m}(r)` is the model, :math:`w(r)`
    is a user-defined weighting function and :math:`p(r) \in [0,2]` is a parameter which imposes
    sparseness throughout the recovered model. Sharper boundaries are recovered in regions
    where :math:`p(r)` is small. If the same level of sparseness is being imposed everywhere,
    the exponent becomes a constant.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discrete approximation for the regularization
    function (objective function) is expressed in linear form as:

    .. math::
        \phi (\mathbf{m}) = \sum_i
        \tilde{w}_i \, \Bigg | \, \frac{\partial |\vec{m}_i|}{\partial x} \, \Bigg |^{p_i}

    where :math:`\vec{m}_i` is the vector defined for mesh cell :math:`i`.
    :math:`\tilde{w}_i` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply user-defined weighting.
    :math:`p_i \in \mathbf{p}` define the norm for each cell (set using `norm`).

    It is impractical to work with the general form directly, as its derivatives with respect
    to the model are non-linear and discontinuous. Instead, the iteratively re-weighted
    least-squares (IRLS) approach is used to approximate the sparse norm by iteratively solving
    a set of convex least-squares problems. For IRLS iteration :math:`k`, we define:

    .. math::
        \phi \big (\mathbf{m}^{(k)} \big )
        = \sum_i
        \tilde{w}_i \, \left | \, \frac{\partial \big | \vec{m}_i^{(k)} \big | }{\partial x} \right |^{p_i}
        \approx \sum_i \tilde{w}_i \, r_i^{(k)}
        \left | \, \frac{\partial \big | \vec{m}_i^{(k)} \big | }{\partial x} \right |^2

    where the IRLS weight :math:`r_i` for iteration :math:`k` is given by:

    .. math::
        r_i^{(k)} = \Bigg [ \Bigg ( \frac{\partial \big | \vec{m}_i^{(k-1)} \big | }{\partial x} \Bigg )^2 +
        \epsilon^2 \; \Bigg ]^{{p_i}/2 - 1}

    and :math:`\epsilon` is a small constant added for stability (set using `irls_threshold`).

    The global set of model parameters :math:`\mathbf{m}` defined at cell centers is ordered according
    to its primary (:math:`p`), secondary (:math:`s`) and tertiary (:math:`t`) directions as follows:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m}_p \\ \mathbf{m}_s \\ \mathbf{m}_t \end{bmatrix}

    We define the amplitudes of the vectors for all cells as:

    .. math::
        \mathbf{\bar{m}} = \Big [ \, \mathbf{m}_p^2 + \mathbf{m}_s^2 + \mathbf{m}_t^2 \Big ]^{1/2}

    The objective function for IRLS iteration :math:`k` is given by:

    .. math::
        \phi \big ( \mathbf{m}^{(k)} \big ) \approx \Big \| \,
        \mathbf{W}^{(k)} \, \mathbf{G_x} \, \mathbf{\bar{m}}^{(k)} \Big \|^2

    where

        - :math:`\bar{\mathbf{m}}^{(k)}` are the discrete vector model amplitudes at iteration :math:`k`,
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
        \mathbf{W}^{(k)} \, \mathbf{G_x} \, \mathbf{\bar{m}}^{(k)} \Big \|^2

    where

    .. math::
        \mathbf{\bar{m}} = \bigg (
        \Big [ \mathbf{m}_p - \mathbf{m}_p^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_s - \mathbf{m}_s^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_t - \mathbf{m}_t^{(ref)} \Big ]^2 \bigg )^{1/2}

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
    """

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        nC = self.regularization_mesh.nC
        nF = getattr(
            self.regularization_mesh, "aveCC2F{}".format(self.orientation)
        ).shape[0]
        return [
            (nF,),
            (self.n_comp * nF,),
            (nF, self.n_comp),
            (nC,),
            (self.n_comp * nC,),
            (nC, self.n_comp),
        ]

    def f_m(self, m):
        r"""Evaluate the regularization kernel function.

        For first-order smoothness regularization in the x-direction,
        the regularization kernel function is given by:

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{G_x \, \bar{m}}

        where :math:`\mathbf{G_x}` is the partial cell gradient operator along the x-direction
        (i.e. x-derivative), and

        .. math::
            \mathbf{f_m}(\mathbf{m}) = \mathbf{\bar{m}} = \bigg (
            \Big [ \mathbf{m}_p - \mathbf{m}_p^{(ref)} \Big ]^2 +
            \Big [ \mathbf{m}_s - \mathbf{m}_s^{(ref)} \Big ]^2 +
            \Big [ \mathbf{m}_t - \mathbf{m}_t^{(ref)} \Big ]^2 \bigg )^{1/2}

        The global set of model parameters :math:`\mathbf{m}` defined at cell centers is
        ordered according to its primary (:math:`p`), secondary (:math:`s`) and tertiary (:math:`t`)
        directions as follows:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m}_p \\ \mathbf{m}_s \\ \mathbf{m}_t \end{bmatrix}

        Likewise for the reference model vector. The expression has the same form for smoothness
        along y and z.

        Parameters
        ----------
        m : numpy.ndarray
            The model.

        Returns
        -------
        numpy.ndarray
            The regularization kernel function evaluated for the model provided.
        """
        fm = self.cell_gradient * (self.mapping * self._delta_m(m)).reshape(
            (self.regularization_mesh.nC, self.n_comp), order="F"
        )

        return np.linalg.norm(fm, axis=1)

    def f_m_deriv(self, m) -> csr_matrix:
        r"""Derivative of the regularization kernel function.

        For first-order smoothness regularization in the x-direction, the derivative of the
        regularization kernel function with respect to the model is given by:

        .. math::
            \frac{\partial \mathbf{f_m}}{\partial \mathbf{m}} =
            \begin{bmatrix} \mathbf{G_x} & \mathbf{0} \\ \mathbf{0} & \mathbf{G_x} \end{bmatrix}

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
        """
        return sp.block_diag([self.cell_gradient] * self.n_comp) @ self.mapping.deriv(
            self._delta_m(m)
        )

    @property
    def W(self):
        ### Inherited
        if getattr(self, "_W", None) is None:
            average_cell_2_face = getattr(
                self.regularization_mesh, "aveCC2F{}".format(self.orientation)
            )
            nC = self.regularization_mesh.nC
            nF = average_cell_2_face.shape[0]
            weights = 1.0
            for values in self._weights.values():
                if values.shape[0] == nC:
                    values = average_cell_2_face * values
                elif not values.shape == (nF,):
                    values = np.linalg.norm(
                        values.reshape((-1, self.n_comp), order="F"), axis=1
                    )
                    if values.size == nC:
                        values = average_cell_2_face * values

                weights *= values

            self._W = sp.diags(np.sqrt(weights), format="csr")

        return self._W


class VectorAmplitude(Sparse):
    r"""Sparse vector amplitude regularization.

    Apply vector amplitude regularization for recovering compact and/or blocky structures
    using a weighted sum of :class:`AmplitudeSmallness` and :class:`AmplitudeSmoothnessFirstOrder`
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
    Sparse vector amplitude regularization can be defined by a weighted sum of
    :class:`AmplitudeSmallness`  and :class:`AmplitudeSmoothnessFirstOrder`
    regularization functions. This corresponds to a model objective function
    :math:`\phi_m (m)` of the form:

    .. math::
        \phi_m (m) = \alpha_s \int_\Omega \, w(r)
        \Big | \, \vec{m}(r) - \vec{m}^{(ref)}(r) \, \Big |^{p_s(r)} \, dv
        + \sum_{j=x,y,z} \alpha_j \int_\Omega \, w(r)
        \Bigg | \, \frac{\partial |\vec{m}|}{\partial \xi_j} \, \bigg |^{p_j(r)} \, dv

    where :math:`\vec{m}(r)` is the model, :math:`\vec{m}^{(ref)}(r)` is the reference model,
    and :math:`w(r)` is a user-defined weighting function applied to all terms.
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
    on the mesh such that:

    .. math::
        f_i = \begin{cases}
        | \, \vec{m}_i \, | \;\;\;\;\;\;\; (no \; reference \; model)\\
        | \, \vec{m}_i - \vec{m}_i^{(ref)} \, | \;\;\;\; (reference \; model)
        \end{cases}

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

    The global set of model parameters :math:`\mathbf{m}` defined at cell centers is ordered according
    to its primary (:math:`p`), secondary (:math:`s`) and tertiary (:math:`t`) directions as follows:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m}_p \\ \mathbf{m}_s \\ \mathbf{m}_t \end{bmatrix}

    The objective function for IRLS iteration :math:`k` can be expressed as a weighted sum of
    objective functions of the form:

    .. math::
        \phi_m (\mathbf{m}) = \alpha_s
        \Big \| \, \mathbf{W_s}^{\! (k)} \, \Delta \mathbf{\bar{m}} \, \Big \|^2
        + \sum_{j=x,y,z} \alpha_j \Big \| \, \mathbf{W_j}^{\! (k)} \mathbf{G_j \, \bar{m}} \, \Big \|^2

    where

    .. math::
        \Delta \mathbf{\bar{m}} = \bigg (
        \Big [ \mathbf{m}_p - \mathbf{m}_p^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_s - \mathbf{m}_s^{(ref)} \Big ]^2 +
        \Big [ \mathbf{m}_t - \mathbf{m}_t^{(ref)} \Big ]^2 \bigg )^{1/2}

    and

    .. math::
        \mathbf{\bar{m}} = \Big [ \, \mathbf{m}_p^2 + \mathbf{m}_s^2 + \mathbf{m}_t^2 \Big ]^{1/2}

    :math:`\mathbf{G_x, \, G_y, \; G_z}` are partial cell gradients operators along x, y and z, and
    :math:`\mathbf{W_s, \, W_x, \, W_y, \; W_z}` are the weighting matrices for iteration :math:`k`.
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

    >>> reg = AmplitudeVector(mesh, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    **Reference model in smoothness:**

    Gradients/interfaces within a discrete reference model can be preserved by including the
    reference model the smoothness regularization. In this case,
    the objective function becomes:

    .. math::
        \phi_m (\mathbf{m}) = \alpha_s
        \Big \| \, \mathbf{W_s}^{\! (k)} \, \Delta \mathbf{\bar{m}} \, \Big \|^2
        + \sum_{j=x,y,z} \alpha_j \Big \| \, \mathbf{W_j}^{\! (k)} \mathbf{G_j \, \Delta \bar{m}} \, \Big \|^2

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
        mapping=None,
        active_cells=None,
        **kwargs,
    ):
        if not isinstance(mesh, (BaseMesh, RegularizationMesh)):
            raise TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )

        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        self._regularization_mesh = mesh

        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        objfcts = [
            AmplitudeSmallness(mesh=self.regularization_mesh, mapping=mapping),
            AmplitudeSmoothnessFirstOrder(
                mesh=self.regularization_mesh, orientation="x", mapping=mapping
            ),
        ]

        if mesh.dim > 1:
            objfcts.append(
                AmplitudeSmoothnessFirstOrder(
                    mesh=self.regularization_mesh, orientation="y", mapping=mapping
                )
            )

        if mesh.dim > 2:
            objfcts.append(
                AmplitudeSmoothnessFirstOrder(
                    mesh=self.regularization_mesh, orientation="z", mapping=mapping
                )
            )

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            mapping=mapping,
            **kwargs,
        )
