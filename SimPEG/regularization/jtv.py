import numpy as np
import scipy.sparse as sp

from .base import BaseSimilarityMeasure


###############################################################################
#                                                                             #
#                            Joint Total Variation                            #
#                                                                             #
###############################################################################


class JointTotalVariation(BaseSimilarityMeasure):
    r"""
    The joint total variation constraint for joint inversions.

    ..math::
        \phi_sim(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} V_i\sqrt{|
        \nabla \mathbf{m_1}_i|^2 +|\nabla \mathbf{m_2}_i|^2}
    """

    # reset this here to clear out the properties attribute
    cell_weights = None

    eps = 1e-8

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)

        regmesh = self.regularization_mesh
        self._G = regmesh.cell_gradient
        vsq = regmesh.vol ** 2
        self._Av = sp.diags(vsq) * regmesh.average_face_to_cell

    def __call__(self, model):
        """
        Computes the sum of all joint total variation values.

        Parameters
        ----------
        model : numpy.ndarray
            stacked array of individual models np.r_[model1, model2,...]

        Returns
        -------
        float
            Tthe computed value of the joint total variation term.
        """
        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        v2 = self.regularization_mesh.vol ** 2
        g_m1 = G @ m1
        g_m2 = G @ m2

        g2 = g_m1 ** 2 + g_m2 ** 2
        Av_g = Av @ g2
        sq = np.sqrt(Av_g + self.eps * v2)
        return np.sum(sq)

    def deriv(self, model):
        """
        Computes the derivative of the joint total variation.

        Parameters
        ----------
        model : numpy.ndarray
            stacked array of individual models np.r_[model1, model2,...]

        Returns
        -------
        numpy.ndarray
            The gradient of joint total variatio  with respect to the model
        """
        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        v2 = self.regularization_mesh.vol ** 2
        g_m1 = G @ m1
        g_m2 = G @ m2

        g2 = g_m1 ** 2 + g_m2 ** 2
        Av_g = Av @ g2
        sq = np.sqrt(Av_g + self.eps * v2)
        mid = Av.T @ (1 / sq)

        return np.r_[G.T @ (mid * g_m1), G.T @ (mid * g_m2)]

    def deriv2(self, model, v=None):
        """
        Computes the Hessian of the joint total variation.

        Parameters
        ----------
        model : numpy.ndarray
            Stacked array of individual models
        v : numpy.ndarray, optional
            An array to multiply the Hessian by.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csr_matrix
            The Hessian of joint total variation with respect to the model times a
            vector or the full Hessian if `v` is `None`.
        """
        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        v2 = self.regularization_mesh.vol ** 2
        g_m1 = G @ m1
        g_m2 = G @ m2

        g2 = g_m1 ** 2 + g_m2 ** 2
        Av_g = Av @ g2
        sq = np.sqrt(Av_g + self.eps * v2)
        mid = Av.T @ (1 / sq)

        if v is not None:
            v1, v2 = self.wire_map * v
            g_v1 = G @ v1
            g_v2 = G @ v2

            p1 = G.T @ (mid * g_v1 - g_m1 * (Av.T @ ((Av @ (g_m1 * g_v1)) / sq ** 3)))
            p2 = G.T @ (mid * g_v2 - g_m2 * (Av.T @ ((Av @ (g_m2 * g_v2)) / sq ** 3)))

            p1 -= G.T @ (g_m1 * (Av.T @ ((Av @ (g_m2 * g_v2)) / sq ** 3)))
            p2 -= G.T @ (g_m2 * (Av.T @ ((Av @ (g_m1 * g_v1)) / sq ** 3)))

            return np.r_[p1, p2]
        else:
            A = (
                G.T
                @ (
                    sp.diags(mid)
                    - sp.diags(g_m1)
                    @ Av.T
                    @ sp.diags(1 / (sq ** 3))
                    @ Av
                    @ sp.diags(g_m1)
                )
                @ G
            )
            C = (
                G.T
                @ (
                    sp.diags(mid)
                    - sp.diags(g_m2)
                    @ Av.T
                    @ sp.diags(1 / (sq ** 3))
                    @ Av
                    @ sp.diags(g_m2)
                )
                @ G
            )

            B = (
                -G.T
                @ sp.diags(g_m1)
                @ Av.T
                @ sp.diags(1 / (sq ** 3))
                @ Av
                @ sp.diags(g_m2)
                @ G
            )
            BT = B.T

            return sp.bmat([[A, B], [BT, C]], format="csr")
