import numpy as np
import scipy.sparse as sp

from .base import BaseSimilarityMeasure


###############################################################################
#                                                                             #
#                            Joint Total Variation                            #
#                                                                             #
###############################################################################


class JointTotalVariation(BaseSimilarityMeasure):
    r"""Joint total variation regularization for joint inversion.

    ``JointTotalVariation`` regularization aims to ensure non-zero gradients in the recovered
    model to occur at the same locations for all physical property distributions.
    It assumes structures within each physical property distribution are sparse and
    correlated with one another.

    Parameters
    ----------
    mesh : simpeg.regularization.RegularizationMesh, discretize.base.BaseMesh
        Mesh on which the regularization is discretized. This is not necessarily
        the same as the mesh on which the simulation is defined.
    active_cells : None, (n_cells, ) numpy.ndarray of bool
        Boolean array defining the set of :py:class:`~.regularization.RegularizationMesh`
        cells that are active in the inversion. If ``None``, all cells are active.
    wire_map : simpeg.maps.Wires
        Wire map connecting physical properties defined on active cells of the
        :class:`RegularizationMesh`` to the entire model.
    reference_model : None, (n_param, ) numpy.ndarray
        Reference model. If ``None``, the reference model in the inversion is set to
        the starting model.
    units : None, str
        Units for the model parameters. Some regularization classes behave
        differently depending on the units; e.g. 'radian'.
    weights : None, dict
        Weight multipliers to customize the least-squares function. Each key points to a (n_cells, )
        numpy.ndarray that is defined on the :py:class:`~.regularization.RegularizationMesh`.
    eps : float
        Needs documentation!!!

    Notes
    -----
    Consider the case where the model is comprised of two physical properties
    :math:`m_1` and :math:`m_2`. Here, we define the regularization
    function (objective function) for joint total variation as
    (`Haber and Gazit, 2013 <https://link.springer.com/article/10.1007/s10712-013-9232-4>`__):

    .. math::
        \phi (m_1, m_2) = \int_\Omega \, w(r) \,
        \Big [ \, \big | \nabla m_1 \big |^2 \, + \, \big | \nabla m_2 \big |^2 \, \Big ]^{1/2} \, dv

    where :math:`w(r)` is a user-defined weighting function.

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is given by:

    .. math::
        \phi (m_1, m_2) \approx \sum_i \tilde{w}_i \, \bigg [ \,
        \Big | (\nabla m_1)_i \Big |^2 \, + \, \Big | (\nabla m_2)_i \Big |^2 \, \bigg ]^{1/2}

    where :math:`(\nabla m_1)_i` are the gradients of property :math:`m_1` defined on the mesh and
    :math:`\tilde{w}_i \in \mathbf{\tilde{w}}` are amalgamated weighting constants that 1) account
    for cell dimensions in the discretization and 2) apply any user-defined weighting.

    In practice, we define the model :math:`\mathbf{m}` as a discrete
    vector of the form:

    .. math::
        \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

    where :math:`\mathbf{m_1}` and :math:`\mathbf{m_2}` are the discrete representations
    of the respective physical properties on the mesh. The discrete regularization function
    is therefore equivalent to an objective function of the form:

    .. math::
        \phi (\mathbf{m}) = \mathbf{e}^T \Bigg ( \,
        \mathbf{W \, A} \bigg [ \sum_k (\mathbf{G \, m_k})^2 \bigg ] \; + \; \epsilon \mathbf{v}^2
        \, \Bigg )^{1/2}

    where exponents are computed elementwise,

        - :math:`\mathbf{e}` is a vector of 1s,
        - :math:`\mathbf{W}` is the weighting matrix for joint total variation regularization,
        - :math:`\mathbf{A}` averages vectors from faces to cell centers,
        - :math:`\mathbf{G}` is the cell gradient operator (cell centers to faces),
        - :math:`\mathbf{v}` are the cell volumes, and
        - :math:`\epsilon` is a constant added for continuous differentiability (set with the `eps` property),

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v}` are the cell volumes.
    The weighting matrix used to apply weights within the regularization is given by:

    .. math::
        \boldsymbol{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^2 \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = JointTotalVariation(
    >>>     mesh, wire_map, weights={'weights_1': array_1, 'weights_2': array_2}
    >>> )

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

    """

    def __init__(self, mesh, wire_map, eps=1e-8, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)
        self.eps = eps

        self._G = self.regularization_mesh.cell_gradient

    @property
    def W(self):
        r"""Weighting matrix for joint total variation regularization.

        Returns the weighting matrix for the discrete regularization function. To see how the
        weighting matrix is constructed, see the *Notes* section for the :class:`JointTotalVariation`
        regularization class.

        Returns
        -------
        scipy.sparse.csr_matrix
            The weighting matrix applied in the regularization.
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = (
                sp.diags(weights**2) * self.regularization_mesh.average_face_to_cell
            )
        return self._W

    @property
    def wire_map(self):
        # Docs inherited from BaseSimilarityMeasure
        return self._wire_map

    @wire_map.setter
    def wire_map(self, wires):
        n = self.regularization_mesh.nC
        maps = wires.maps
        for _, mapping in maps:
            map_n = mapping.shape[0]
            if n != map_n:
                raise ValueError(
                    f"All mapping outputs must match the number of cells in "
                    f"the regularization mesh! Got {n} and {map_n}"
                )
        self._wire_map = wires

    def __call__(self, model):
        """Evaluate the joint total variation regularization function for the model provided.

        See the *Notes* section of the documentation for the :class:`JointTotalVariation` class
        for a full description of the regularization function.

        Parameters
        ----------
        m : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.

        Returns
        -------
        float
            The regularization function evaluated for the model provided.
        """
        W = self.W
        G = self._G
        v2 = self.regularization_mesh.vol**2
        g2 = 0
        for m in self.wire_map * model:
            g_m = G @ m
            g2 += g_m**2
        W_g = W @ g2
        sq = np.sqrt(W_g + self.eps * v2)
        return np.sum(sq)

    def deriv(self, model):
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters;
        i.e. the gradient. For a model :math:`\mathbf{m}` consisting of multiple physical properties
        :math:`\mathbf{m_1}, \; \mathbf{m_2}, \; ...` such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \\ \vdots \end{bmatrix}

        The gradient has the form:

        .. math::
            \frac{\partial \phi}{\partial \mathbf{m}} =
            \begin{bmatrix} \dfrac{\partial \phi}{\partial \mathbf{m_1}} \\
            \dfrac{\partial \phi}{\partial \mathbf{m_2}} \\ \vdots \end{bmatrix}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Gradient of the regularization function evaluated for the model provided.
        """
        W = self.W
        G = self._G
        g2 = 0
        gs = []
        v2 = self.regularization_mesh.vol**2
        for m in self.wire_map * model:
            g_mi = G @ m
            g2 += g_mi**2
            gs.append(g_mi)
        W_g = W @ g2
        sq = np.sqrt(W_g + self.eps * v2)
        mid = W.T @ (1 / sq)
        ps = []
        for g_mi in gs:
            ps.append(G.T @ (mid * g_mi))
        return np.concatenate(ps)

    def deriv2(self, model, v=None):
        r"""Hessian of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evalutate and returns the second derivative (Hessian) with respect to the model parameters.
        For a model :math:`\mathbf{m}` consisting of multiple physical properties
        :math:`\mathbf{m_1}, \; \mathbf{m_2}, \; ...` such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \\ \vdots \end{bmatrix}

        The Hessian has the form:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} =
            \begin{bmatrix}
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1}^2} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1} \partial \mathbf{m_2}} &
            \cdots \\
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2} \partial \mathbf{m_1}} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2}^2} & \; \\
            \vdots & \; & \ddots
            \end{bmatrix}

        When a vector :math:`(\mathbf{v})` is supplied, the method returns the Hessian
        times the vector:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.
        v : numpy.ndarray, optional
            An array to multiply the Hessian by.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            Hessian of the regularization function evaluated for the model provided.
            The Hessian of joint total variation with respect to the model times a
            vector or the full Hessian if `v` is `None`.
        """
        W = self.W
        G = self._G
        v2 = self.regularization_mesh.vol**2
        gs = []
        g2 = 0
        for m in self.wire_map * model:
            g_m = G @ m
            g2 += g_m**2
            gs.append(g_m)

        W_g = W @ g2
        sq = np.sqrt(W_g + self.eps * v2)
        mid = W.T @ (1 / sq)

        if v is not None:
            g_vs = []
            tmp_sum = 0
            for vi, g_i in zip(self.wire_map * v, gs):
                g_vi = G @ vi
                tmp_sum += W.T @ ((W @ (g_i * g_vi)) / sq**3)
                g_vs.append(g_vi)
            ps = []
            for g_vi, g_i in zip(g_vs, gs):
                ps.append(G.T @ (mid * g_vi - g_i * tmp_sum))
            return np.concatenate(ps)
        else:
            Pieces = []
            Diags = []
            SQ = sp.diags(sq**-1.5)
            diag_block = G.T @ sp.diags(mid) @ G
            for g_mi in gs:
                Pieces.append(SQ @ W @ sp.diags(g_mi) @ G)
                Diags.append(diag_block)
            Row = sp.hstack(Pieces, format="csr")
            Diag = sp.block_diag(Diags, format="csr")
            return Diag - Row.T @ Row
