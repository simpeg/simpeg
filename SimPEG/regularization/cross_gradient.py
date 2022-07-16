import numpy as np
import scipy.sparse as sp
import properties

from .base import BaseSimilarityMeasure


###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseSimilarityMeasure):
    """
    The cross-gradient constraint for joint inversions.

    ..math::
        \\phi_c(\\mathbf{m_1},\\mathbf{m_2}) = \\lambda \\sum_{i=1}^{M} \\|
        \\nabla \\mathbf{m_1}_i \\times \\nabla \\mathbf{m_2}_i \\|^2

    All methods assume that we are working with two models only.

    """

    # These are not fully implemented yet
    # grad_tol = properties.Float(
    #     "tolerance for avoiding the exteremly small gradient amplitude", default=1e-10
    # )
    # normalized = properties.Bool(
    #     "whether to implement normalized cross-gradient", default=False
    # )

    approx_hessian = properties.Bool(
        "whether to use the semi-positive definate approximation for the hessian",
        default=True,
    )

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)

        regmesh = self.regularization_mesh

        if regmesh.mesh.dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")
        self._G = regmesh.cell_gradient
        self._Av = sp.diags(np.sqrt(regmesh.vol)) * regmesh.average_face_to_cell

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
        """
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True

        :rtype: float
        :returns: the computed value of the cross-gradient term.


        ..math::

            \\phi_c(\\mathbf{m_1},\\mathbf{m_2})

            = \\lambda \\sum_{i=1}^{M} \\|\\nabla \\mathbf{m_1}_i \\times \\nabla \\mathbf{m_2}_i \\|^2

            = \\sum_{i=1}^{M} \\|\\nabla \\mathbf{m_1}_i\\|^2 \\ast \\|\\nabla \\mathbf{m_2}_i\\|^2
                - (\\nabla \\mathbf{m_1}_i \\cdot \\nabla \\mathbf{m_2}_i )^2

            = \\|\\phi_{cx}\\|^2 + \\|\\phi_{cy}\\|^2 + \\|\\phi_{cz}\\|^2 (optional strategy, not used in this script)


        """
        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        g_m1 = G @ m1
        g_m2 = G @ m2
        return 0.5 * np.sum(
            (Av @ g_m1 ** 2) * (Av @ g_m2 ** 2) - (Av @ (g_m1 * g_m2)) ** 2
        )

    def deriv(self, model):
        """
        Computes the Jacobian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2

        """
        m1, m2 = self.wire_map * model

        Av = self._Av
        G = self._G
        g_m1 = G @ m1
        g_m2 = G @ m2

        return np.r_[
            (((Av @ g_m2 ** 2) @ Av) * g_m1) @ G
            - (((Av @ (g_m1 * g_m2)) @ Av) * g_m2) @ G,
            (((Av @ g_m1 ** 2) @ Av) * g_m2) @ G
            - (((Av @ (g_m1 * g_m2)) @ Av) * g_m1) @ G,
        ]

    def deriv2(self, model, v=None):
        """
        Computes the Hessian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian

        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix if v is None
                Hessian multiplied by vector if v is not No

        """
        m1, m2 = self.wire_map * model

        Av = self._Av
        G = self._G

        g_m1 = G @ m1
        g_m2 = G @ m2

        if v is None:
            A = (
                G.T
                @ (
                    sp.diags(Av.T @ (Av @ g_m2 ** 2))
                    - sp.diags(g_m2) @ Av.T @ Av @ sp.diags(g_m2)
                )
                @ G
            )

            C = (
                G.T
                @ (
                    sp.diags(Av.T @ (Av @ g_m1 ** 2))
                    - sp.diags(g_m1) @ Av.T @ Av @ sp.diags(g_m1)
                )
                @ G
            )

            B = None
            BT = None
            if not self.approx_hessian:
                # d_m1_d_m2
                B = (
                    G.T
                    @ (
                        2 * sp.diags(g_m1) @ Av.T @ Av @ sp.diags(g_m2)
                        - sp.diags(g_m2) @ Av.T @ Av @ sp.diags(g_m1)
                        - sp.diags(Av.T @ Av @ (g_m1 * g_m2))
                    )
                    @ G
                )
                BT = B.T

            return sp.bmat([[A, B], [BT, C]], format="csr")
        else:
            v1, v2 = self.wire_map * v

            Gv1 = G @ v1
            Gv2 = G @ v2

            p1 = G.T @ (
                (Av.T @ (Av @ g_m2 ** 2)) * Gv1 - g_m2 * (Av.T @ (Av @ (g_m2 * Gv1)))
            )
            p2 = G.T @ (
                (Av.T @ (Av @ g_m1 ** 2)) * Gv2 - g_m1 * (Av.T @ (Av @ (g_m1 * Gv2)))
            )

            if not self.approx_hessian:
                p1 += G.T @ (
                    2 * g_m1 * (Av.T @ (Av @ (g_m2 * Gv2)))
                    - g_m2 * (Av.T @ (Av @ (g_m1 * Gv2)))
                    - (Av.T @ (Av @ (g_m1 * g_m2))) * Gv2
                )

                p2 += G.T @ (
                    2 * g_m2 * (Av.T @ (Av @ (g_m1 * Gv1)))
                    - g_m1 * (Av.T @ (Av @ (g_m2 * Gv1)))
                    - (Av.T @ (Av @ (g_m2 * g_m1))) * Gv1
                )
            return np.r_[p1, p2]
