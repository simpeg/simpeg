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

    ..math ::
        \phi_sim(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} V_i\sqrt{|
        \nabla \mathbf{m_1}_i|^2 +|\nabla \mathbf{m_2}_i|^2}
    """

    def __init__(self, mesh, wire_map, eps=1e-8, **kwargs):
        super().__init__(mesh, wire_map=wire_map, **kwargs)
        self.set_weights(volume=self.regularization_mesh.vol)
        self.eps = eps

        self._G = self.regularization_mesh.cell_gradient

    @property
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = np.prod(list(self._weights.values()), axis=0)
            self._W = (
                sp.diags(weights**2) * self.regularization_mesh.average_face_to_cell
            )
        return self._W

    @property
    def wire_map(self):
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
        """
        Computes the sum of all joint total variation values.

        Parameters
        ----------
        model : numpy.ndarray
            stacked array of individual models np.r_[model1, model2,...]

        Returns
        -------
        float
            The computed value of the joint total variation term.
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
        """
        Computes the derivative of the joint total variation.

        Parameters
        ----------
        model : numpy.ndarray
            stacked array of individual models np.r_[model1, model2,...]

        Returns
        -------
        numpy.ndarray
            The gradient of joint total variation  with respect to the model
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
