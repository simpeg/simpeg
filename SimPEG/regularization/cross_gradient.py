import numpy as np
import scipy.sparse as sp
import properties

from .base import BaseCoupling


###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseCoupling):
    """
    The cross-gradient constraint for joint inversions.

    ..math::
        \\phi_c(\\mathbf{m_1},\\mathbf{m_2}) = \\lambda \\sum_{i=1}^{M} \\|
        \\nabla \\mathbf{m_1}_i \\times \\nabla \\mathbf{m_2}_i \\|^2

    All methods assume that we are working with two models only.

    """

    grad_tol = properties.Float(
        "tolerance for avoiding the exteremly small gradient amplitude", default=1e-10
    )
    normalized = properties.Bool(
        "whether to implement normalized cross-gradient", default=False
    )

    approx_hessian = properties.Bool(
        "whether to use the semi-positive definate approximation for the hessian",
        default=True,
    )

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)

        regmesh = self.regmesh

        self._dim = regmesh.mesh.dim
        if self._dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")

        Av = [
            regmesh.aveFx2CC,
            regmesh.aveFy2CC,
        ]
        G = [regmesh.cellDiffx, regmesh.cellDiffy]
        if regmesh.mesh.dim == 3:
            Av.append(regmesh.aveFz2CC)
            G.append(regmesh.cellDiffz)
        self._G = sp.vstack(G)
        self._Av = sp.hstack(Av)

    def _calculate_gradient(self, model):
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
        gradient = (self.Av @ (self._G @ model)).reshape((-1, self.dim), order="F")

        if self.normalized:
            norms = np.linalg.norm(gradient, axis=-1)
            ind = norms >= self.grad_tol
            gradient[ind] /= norms[~ind, None]
            gradient[~ind] = 0.0
            # set gradient to 0 if amplitude of gradient is extremely small

        return gradient

    def calculate_cross_gradient(self, m1, m2):
        """
        Calculates the cross-gradients of two models at each cell center.

        :param numpy.ndarray m1: model1
        :param numpy.ndarray m2: model2
        :param bool normalized: normalizes gradients if True

        :rtype: numpy.ndarray
        :returns: array where at each location, we've computed the cross-product
                  of the gradients of two models.

        """
        # Compute the gradients and concatenate components.
        grad_list_m1 = self._calculate_gradient(m1)
        grad_list_m2 = self._calculate_gradient(m2)

        # for each model cell, compute the cross product of the gradient vectors.
        if self._dim == 3:
            cross_prod_vector = np.cross(grad_list_m1, grad_list_m2)
            cross_prod = np.linalg.norm(cross_prod_vector, axis=-1)
        else:
            cross_prod = np.cross(grad_list_m1, grad_list_m2)

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

            \phi_c(\mathbf{m_1},\mathbf{m_2})

            = \lambda \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

            = \sum_{i=1}^{M} \|\nabla \mathbf{m_1}_i\|^2 \ast \|\nabla \mathbf{m_2}_i\|^2
                - (\nabla \mathbf{m_1}_i \cdot \nabla \mathbf{m_2}_i )^2

            = \|\phi_{cx}\|^2 + \|\phi_{cy}\|^2 + \|\phi_{cz}\|^2 (optional strategy, not used in this script)


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


###############################################################################
#                                                                             #
#               Linear petrophysical relationship constraint                  #
#                                                                             #
###############################################################################
