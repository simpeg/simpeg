import numpy as np
import scipy.sparse as sp
import properties

from .base import BaseSimilarityMeasure


###############################################################################
#                                                                             #
#                                Cross-Gradient                               #
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

    # These are not fully implemented yet
    # grad_tol = properties.Float(
    #     "tolerance for avoiding the exteremly small gradient amplitude", default=1e-10
    # )
    # normalized = properties.Bool(
    #     "whether to implement normalized cross-gradient", default=False
    # )
    eps = 1e-8

    approx_hessian = properties.Bool(
        "whether to use the semi-positive definate approximation for the hessian",
        default=True,
    )

    def __init__(self, mesh, wire_map, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)

        regmesh = self.regmesh

        if regmesh.mesh.dim not in (2, 3):
            raise ValueError("Cross-Gradient is only defined for 2D or 3D")
        self._G = regmesh.cell_gradient
        vsq = regmesh.vol ** 2
        self._Av = sp.diags(vsq) * regmesh.average_face_to_cell

    def __call__(self, model):
        """
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]

        :rtype: float
        :returns: the computed value of the joint total variation term.


        """
        m1, m2 = self.wire_map * model
        Av = self._Av
        G = self._G
        v2 = self.regmesh.vol ** 2
        g_m1 = G @ m1
        g_m2 = G @ m2

        g2 = g_m1 ** 2 + g_m2 ** 2
        Av_g = Av @ g2
        sq = np.sqrt(Av_g + self.eps * v2)
        return np.sum(sq)

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
        v2 = self.regmesh.vol ** 2
        g_m1 = G @ m1
        g_m2 = G @ m2

        g2 = g_m1 ** 2 + g_m2 ** 2
        Av_g = Av @ g2
        sq = np.sqrt(Av_g + self.eps * v2)
        mid = Av.T @ (1 / sq)

        return np.r_[G.T @ (mid * g_m1), G.T @ (mid * g_m2)]

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
        v2 = self.regmesh.vol ** 2
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
            if not self.approx_hessian:
                p1 -= G.T @ (g_m1 * (Av.T @ ((Av @ (g_m2 * g_v2)) / sq ** 3)))
                p2 -= G.T @ (g_m2 * (Av.T @ ((Av @ (g_m1 * g_v1)) / sq ** 3)))
            return np.r_[p1, p2]
        else:
            A = (
                G.T @ sp.diags(mid) @ G
                - G.T
                @ sp.diags(g_m1)
                @ Av.T
                @ sp.diags(1 / (sq ** 3))
                @ Av
                @ sp.diags(g_m1)
                @ G
            )
            C = (
                G.T @ sp.diags(mid) @ G
                - G.T
                @ sp.diags(g_m2)
                @ Av.T
                @ sp.diags(1 / (sq ** 3))
                @ Av
                @ sp.diags(g_m2)
                @ G
            )

            B = None
            BT = None
            if not self.approx_hessian:
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
