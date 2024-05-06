import numpy as np
import scipy.sparse as sp

from .base import BaseSimilarityMeasure
from ..utils import validate_type


###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseSimilarityMeasure):
    r"""Cross-gradient regularization for joint inversion.

    ``CrossGradient`` regularization is used to ensure the location and orientation of non-zero
    gradients in the recovered model are consistent across two physical property distributions.
    For joint inversion involving three or more physical properties, a separate instance of
    ``CrossGradient`` must be created for each physical property pair and added to the total
    regularization as a weighted sum.

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
    approx_hessian : bool
        Whether to use the semi-positive definate approximation for the Hessian.

    Notes
    -----
    Consider the case where the model is comprised of two physical properties
    :math:`m_1` and :math:`m_2`. Here, we define the regularization
    function (objective function) for cross-gradient as
    (`Haber and Gazit, 2013 <https://link.springer.com/article/10.1007/s10712-013-9232-4>`__):

    .. math::
        \phi (m_1, m_2) = \int_\Omega \, w(r) \,
        \Big | \nabla m_1 \, \times \, \nabla m_2 \, \Big |^2 \, dv

    where :math:`w(r)` is a user-defined weighting function.
    Using the identity :math:`| \vec{a} \times \vec{b} |^2 = | \vec{a} |^2 | \vec{b} |^2 - (\vec{a} \cdot \vec{b})^2`,
    the regularization function can be re-expressed as:

    .. math::
        \phi (m_1, m_2) = \int_\Omega \, w(r) \, \Big [ \,
        \big | \nabla m_1 \big |^2 \big | \nabla m_2 \big |^2
        - \big ( \nabla m_1 \, \cdot \, \nabla m_2 \, \big )^2 \Big ] \, dv

    For implementation within SimPEG, the regularization function and its variables
    must be discretized onto a `mesh`. The discretized approximation for the regularization
    function (objective function) is given by:

    .. math::
        \phi (m_1, m_2) \approx \sum_i \tilde{w}_i \, \bigg [
        \Big | (\nabla m_1)_i \Big |^2 \Big | (\nabla m_2)_i \Big |^2
        - \Big [ (\nabla m_1)_i \, \cdot \, (\nabla m_2)_i \, \Big ]^2 \, \bigg ]

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
        \phi (\mathbf{m}) =
     \Big [ \mathbf{W A} \big ( \mathbf{G \, m_1} \big )^2 \Big ]^T
        \Big [ \mathbf{W A} \big ( \mathbf{G \, m_2} \big )^2 \Big ]
        - \bigg \| \mathbf{W A} \Big [ \big ( \mathbf{G \, m_1} \big )
        \odot \big ( \mathbf{G \, m_2} \big ) \Big ] \bigg \|^2

    where exponents are computed elementwise,

        - :math:`\mathbf{G}` is the cell gradient operator (cell centers to faces),
        - :math:`\mathbf{A}` averages vectors from faces to cell centers, and
        - :math:`\mathbf{W}` is the weighting matrix.

    **Custom weights and the weighting matrix:**

    Let :math:`\mathbf{w_1, \; w_2, \; w_3, \; ...}` each represent an optional set of
    custom cell weights. The weighting applied within the objective function is given by:

    .. math::
        \mathbf{\tilde{w}} = \mathbf{v} \odot \prod_j \mathbf{w_j}

    where :math:`\mathbf{v}` are the cell volumes.
    The weighting matrix used to apply weights within the regularization is given by:

    .. math::
        \mathbf{W} = \textrm{diag} \Big ( \, \mathbf{\tilde{w}}^{1/2} \Big )

    Each set of custom cell weights is stored within a ``dict`` as an (n_cells, )
    ``numpy.ndarray``. The weights can be set all at once during instantiation
    with the `weights` keyword argument as follows:

    >>> reg = CrossGradient(mesh, wire_map, weights={'weights_1': array_1, 'weights_2': array_2})

    or set after instantiation using the `set_weights` method:

    >>> reg.set_weights(weights_1=array_1, weights_2=array_2})

    The default weights that account for cell dimensions in the regularization are accessed via:

    >>> reg.get_weights('volume')

    """

    def __init__(self, mesh, wire_map, approx_hessian=True, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        self.approx_hessian = approx_hessian

        regmesh = self.regularization_mesh

        if regmesh.mesh.dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")
        self._G = regmesh.cell_gradient
        self._Av = sp.diags(np.sqrt(regmesh.vol)) * regmesh.average_face_to_cell

    @property
    def approx_hessian(self):
        """Whether to use the semi-positive definate approximation for the Hessian.

        Returns
        -------
        bool
            Whether to use the semi-positive definate approximation for the Hessian.
        """
        return self._approx_hessian

    @approx_hessian.setter
    def approx_hessian(self, value):
        self._approx_hessian = validate_type("approx_hessian", value, bool)

    def _calculate_gradient(self, model, normalized=False, rtol=1e-6):
        """
        Calculate the spatial gradients of the model using central difference.

        Concatenates gradient components into a single array.
        [[x_grad1, y_grad1, z_grad1],
         [x_grad2, y_grad2, z_grad2],
         [x_grad3, y_grad3, z_grad3],...]

        :param numpy.ndarray model: model

        :rtype: numpy.ndarray
        :return: gradient_vector: array where each row represents a model cell,
                 and each column represents a component of the gradient.

        """
        regmesh = self.regularization_mesh
        Avs = [regmesh.aveFx2CC, regmesh.aveFy2CC]
        if regmesh.dim == 3:
            Avs.append(regmesh.aveFz2CC)
        Av = sp.block_diag(Avs)
        gradient = (Av @ (self._G @ model)).reshape((-1, regmesh.dim), order="F")

        if normalized:
            norms = np.linalg.norm(gradient, axis=-1)
            ind = norms <= norms.max() * rtol
            norms[ind] = 1.0
            gradient /= norms[:, None]
            gradient[ind] = 0.0
            # set gradient to 0 if amplitude of gradient is extremely small

        return gradient

    def calculate_cross_gradient(self, model, normalized=False, rtol=1e-6):
        r"""Calculates the magnitudes of the cross-gradient vectors at cell centers.

        Computes and returns a discrete approximation to:

        .. math::
            \big | \, \nabla m_1 \, \times \, \nabla m_2 \, \big |

        at all cell centers where :math:`m_1` and :math:`m_2` define the continuous
        spacial distribution of physical properties 1 and 2.

        Parameters
        ----------
        model : numpy.ndarray
            The input model, which will be automatically separated into the two
            parameters internally.
        normalized : bool, optional
            Whether to normalize the cross-gradients.
        rtol : float, optional
            relative cuttoff for small gradients in the normalization.

        Returns
        -------
        numpy.ndarray
            Magnitudes of the cross-gradient vectors at cell centers.
        """
        m1, m2 = self.wire_map * model
        # Compute the gradients and concatenate components.
        grad_m1 = self._calculate_gradient(m1, normalized=normalized, rtol=rtol)
        grad_m2 = self._calculate_gradient(m2, normalized=normalized, rtol=rtol)

        # for each model cell, compute the cross product of the gradient vectors.
        cross_prod = np.cross(grad_m1, grad_m2)
        if self.regularization_mesh.dim == 3:
            cross_prod = np.linalg.norm(cross_prod, axis=-1)

        return cross_prod

    def __call__(self, model):
        """Evaluate the cross-gradient regularization function for the model provided.

        See the *Notes* section of the documentation for the :class:`CrossGradient` class
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

        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        g_m1 = G @ m1
        g_m2 = G @ m2
        return np.sum((Av @ g_m1**2) * (Av @ g_m2**2) - (Av @ (g_m1 * g_m2)) ** 2)

    def deriv(self, model):
        r"""Gradient of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evaluates and returns the derivative with respect to the model parameters;
        i.e. the gradient. For a model :math:`\mathbf{m}` consisting of two physical properties
        such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        The gradient has the form:

        .. math::
            2 \frac{\partial \phi}{\partial \mathbf{m}} =
            \begin{bmatrix} \dfrac{\partial \phi}{\partial \mathbf{m_1}} \\
            \dfrac{\partial \phi}{\partial \mathbf{m_2}} \end{bmatrix}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.

        Returns
        -------
        (n_param, ) numpy.ndarray
            Gradient of the regularization function evaluated for the model provided.
        """
        m1, m2 = self.wire_map * model

        Av = self._Av
        G = self._G
        g_m1 = G @ m1
        g_m2 = G @ m2

        return (
            2
            * np.r_[
                (((Av @ g_m2**2) @ Av) * g_m1) @ G
                - (((Av @ (g_m1 * g_m2)) @ Av) * g_m2) @ G,
                (((Av @ g_m1**2) @ Av) * g_m2) @ G
                - (((Av @ (g_m1 * g_m2)) @ Av) * g_m1) @ G,
            ]
        )  # factor of 2 from derviative of | grad m1 x grad m2 | ^2

    def deriv2(self, model, v=None):
        r"""Hessian of the regularization function evaluated for the model provided.

        Where :math:`\phi (\mathbf{m})` is the discrete regularization function (objective function),
        this method evalutate and returns the second derivative (Hessian) with respect to the model parameters:
        For a model :math:`\mathbf{m}` consisting of two physical properties such that:

        .. math::
            \mathbf{m} = \begin{bmatrix} \mathbf{m_1} \\ \mathbf{m_2} \end{bmatrix}

        The Hessian has the form:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} =
            \begin{bmatrix}
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1}^2} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_1} \partial \mathbf{m_2}} \\
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2} \partial \mathbf{m_1}} &
            \dfrac{\partial^2 \phi}{\partial \mathbf{m_2}^2}
            \end{bmatrix}

        When a vector :math:`(\mathbf{v})` is supplied, the method returns the Hessian
        times the vector:

        .. math::
            \frac{\partial^2 \phi}{\partial \mathbf{m}^2} \, \mathbf{v}

        Parameters
        ----------
        model : (n_param, ) numpy.ndarray
            The model; a vector array containing all physical properties.
        v : None, (n_param, ) numpy.ndarray (optional)
            A numpy array to model the Hessian by.

        Returns
        -------
        (n_param, n_param) scipy.sparse.csr_matrix | (n_param, ) numpy.ndarray
            If the input argument *v* is ``None``, the Hessian
            for the models provided is returned. If *v* is not ``None``,
            the Hessian multiplied by the vector provided is returned.
        """
        m1, m2 = self.wire_map * model

        Av = self._Av
        G = self._G

        g_m1 = G @ m1
        g_m2 = G @ m2

        d11_mid = Av.T @ (Av @ g_m2**2)
        d12_mid = -(Av.T @ (Av @ (g_m1 * g_m2)))
        d22_mid = Av.T @ (Av @ g_m1**2)

        if v is None:
            D11_mid = sp.diags(d11_mid)
            D12_mid = sp.diags(d12_mid)
            D22_mid = sp.diags(d22_mid)
            if not self.approx_hessian:
                D11_mid = D11_mid - sp.diags(g_m2) @ Av.T @ Av @ sp.diags(g_m2)
                D12_mid = (
                    D12_mid
                    + 2 * sp.diags(g_m1) @ Av.T @ Av @ sp.diags(g_m2)
                    - sp.diags(g_m2) @ Av.T @ Av @ sp.diags(g_m1)
                )
                D22_mid = D22_mid - sp.diags(g_m1) @ Av.T @ Av @ sp.diags(g_m1)
            D11 = G.T @ D11_mid @ G
            D12 = G.T @ D12_mid @ G
            D22 = G.T @ D22_mid @ G

            return 2 * sp.bmat(
                [[D11, D12], [D12.T, D22]], format="csr"
            )  # factor of 2 from derviative of | grad m1 x grad m2 | ^2

        else:
            v1, v2 = self.wire_map * v

            Gv1 = G @ v1
            Gv2 = G @ v2
            p1 = G.T @ (d11_mid * Gv1 + d12_mid * Gv2)
            p2 = G.T @ (d12_mid * Gv1 + d22_mid * Gv2)
            if not self.approx_hessian:
                p1 += G.T @ (
                    -g_m2 * (Av.T @ (Av @ (g_m2 * Gv1)))  # d11*v1 full addition
                    + 2 * g_m1 * (Av.T @ (Av @ (g_m2 * Gv2)))  # d12*v2 full addition
                    - g_m2 * (Av.T @ (Av @ (g_m1 * Gv2)))  # d12*v2 continued
                )

                p2 += G.T @ (
                    -g_m1 * (Av.T @ (Av @ (g_m1 * Gv2)))  # d22*v2 full addition
                    + 2 * g_m2 * (Av.T @ (Av @ (g_m1 * Gv1)))  # d12.T*v1 full addition
                    - g_m1 * (Av.T @ (Av @ (g_m2 * Gv1)))  # d12.T*v1 fcontinued
                )
            return (
                2 * np.r_[p1, p2]
            )  # factor of 2 from derviative of | grad m1 x grad m2 | ^2
