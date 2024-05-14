import numpy as np
import scipy.sparse as sp

from .base import BaseSimilarityMeasure
from ..utils import validate_type
from ..utils.mat_utils import coterminal

###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseSimilarityMeasure):
    r"""
    The cross-gradient constraint for joint inversions.

    ..math::
        \phi_c(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} \|
        \nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

    All methods assume that we are working with two models only.

    """

    def __init__(self, mesh, wire_map, approx_hessian=True, normalize=False, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        self.approx_hessian = approx_hessian
        self._units = ["metric", "metric"]
        self.normalize = normalize
        regmesh = self.regularization_mesh

        if regmesh.mesh.dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")
        self._G = regmesh.cell_gradient
        self._Av = sp.diags(np.sqrt(regmesh.vol)) * regmesh.average_face_to_cell

    @property
    def approx_hessian(self):
        """whether to use the semi-positive definate approximation for the hessian.
        Returns
        -------
        bool
        """
        return self._approx_hessian

    @approx_hessian.setter
    def approx_hessian(self, value):
        self._approx_hessian = validate_type("approx_hessian", value, bool)

    def _model_gradients(self, models):
        """
        Compute gradient on faces
        """
        gradients = []

        for unit, wire in zip(self.units, self.wire_map):
            model = wire * models
            if unit == "radian":
                gradient = []
                components = "xyz" if self.regularization_mesh.dim == 3 else "xy"
                for comp in components:
                    distances = getattr(
                        self.regularization_mesh, f"cell_distances_{comp}"
                    )
                    cell_grad = getattr(
                        self.regularization_mesh, f"cell_gradient_{comp}"
                    )
                    gradient.append(
                        coterminal(cell_grad * model * distances) / distances
                    )

                gradient = np.hstack(gradient) / np.pi
            else:
                gradient = self._G @ model

            gradients.append(gradient)

        return gradients

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

        # Compute the gradients and concatenate components.
        grad_models = self._model_gradients(model)

        gradients = []
        for gradient in grad_models:
            gradient = (Av @ (gradient)).reshape((-1, regmesh.dim), order="F")

            if normalized:
                norms = np.linalg.norm(gradient, axis=-1)
                ind = norms <= norms.max() * rtol
                norms[ind] = 1.0
                gradient /= norms[:, None]
                gradient[ind] = 0.0
                # set gradient to 0 if amplitude of gradient is extremely small
            gradients.append(gradient)

        return gradients

    def calculate_cross_gradient(self, model, normalized=False, rtol=1e-6):
        """
        Calculates the cross-gradients of the models at each cell center.

        Parameters
        ----------
        model : numpy.ndarray
            The input model, which will be automatically separated into the two
            parameters internally
        normalized : bool, optional
            Whether to normalize the gradient
        rtol : float, optional
            relative cuttoff for small gradients in the normalization

        Returns
        -------
        cross_grad : numpy.ndarray
            The norm of the cross gradient vector in each active cell.
        """
        # Compute the gradients and concatenate components.
        grad_m1, grad_m2 = self._calculate_gradient(
            model, normalized=normalized, rtol=rtol
        )

        # for each model cell, compute the cross product of the gradient vectors.
        cross_prod = np.cross(grad_m1, grad_m2)
        if self.regularization_mesh.dim == 3:
            cross_prod = np.linalg.norm(cross_prod, axis=-1)

        return cross_prod

    def __call__(self, model):
        r"""
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True

        :rtype: float
        :returns: the computed value of the cross-gradient term.


        ..math::

            \phi_c(\mathbf{m_1},\mathbf{m_2})
            = \lambda \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2
            = \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i\|^2 \ast \|\nabla \mathbf{m_2}_i\|^2
                - (\nabla \mathbf{m_1}_i \cdot \nabla \mathbf{m_2}_i )^2
            = \|\phi_{cx}\|^2 + \|\phi_{cy}\|^2 + \|\phi_{cz}\|^2

        (optional strategy, not used in this script)

        """
        Av = self._Av
        G = self._G
        g_m1, g_m2 = self._model_gradients(model)

        return 0.5 * np.sum(
            np.sum((Av @ g_m1**2) * (Av @ g_m2**2) - (Av @ (g_m1 * g_m2)) ** 2)
        )

    def deriv(self, model):
        """
        Computes the Jacobian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2

        """
        Av = self._Av
        G = self._G
        g_m1, g_m2 = self._model_gradients(model)

        deriv = np.r_[
            (((Av @ g_m2**2) @ Av) * g_m1) @ G
            - (((Av @ (g_m1 * g_m2)) @ Av) * g_m2) @ G,
            (((Av @ g_m1**2) @ Av) * g_m2) @ G
            - (((Av @ (g_m1 * g_m2)) @ Av) * g_m1) @ G,
        ]
        n_cells = self.regularization_mesh.nC
        max_derivs = np.r_[
            np.ones(n_cells) * np.abs(deriv[:n_cells]).max(),
            np.ones(n_cells) * np.abs(deriv[n_cells:]).max(),
        ]

        return self.wire_map_deriv.T * deriv

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
        Av = self._Av
        G = self._G

        g_m1, g_m2 = self._model_gradients(model)

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

            return (
                self.wire_map_deriv.T
                @ sp.bmat([[D11, D12], [D12.T, D22]], format="csr")
                * self.wire_map_deriv
            )  # factor of 2 from derviative of | grad m1 x grad m2 | ^2

        else:
            v1, v2 = (wire * v for wire in self.wire_map)

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
            return self.wire_map_deriv.T * np.r_[p1, p2]
