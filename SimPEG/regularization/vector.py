from __future__ import annotations

# Regularizations for vector models.

import scipy.sparse as sp
import numpy as np
from .base import BaseRegularization, Smallness


class BaseVectorRegularization(BaseRegularization):
    """The regularizers work on models where each value is a vector."""

    @property
    def _weights_shapes(self) -> tuple[int] | str:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        mesh = self.regularization_mesh
        return [(mesh.nC,), (mesh.dim * mesh.nC,), (mesh.nC, mesh.dim)]


class CrossReferenceRegularization(Smallness, BaseVectorRegularization):
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
        return self._ref_dir

    @ref_dir.setter
    def ref_dir(self, value):
        mesh = self.regularization_mesh
        nC = mesh.nC
        value = np.asarray(value)
        if value.shape != (nC, mesh.dim):
            if value.shape == (mesh.dim,):
                # expand it out for each mesh cell
                ref_dir = np.tile(value, (nC, 1))
            else:
                raise ValueError(f"ref_dir must be shape {(nC, mesh.dim)}")
        self._ref_dir = ref_dir

        R0 = sp.diags(ref_dir[:, 0])
        R1 = sp.diags(ref_dir[:, 1])
        if ref_dir.shape[1] == 2:
            X = sp.bmat([[R1, -R0]])
        elif ref_dir.shape[1] == 3:
            Z = sp.csr_matrix((nC, nC))
            R2 = sp.diags(ref_dir[:, 2])
            X = sp.bmat(
                [
                    [Z, R2, -R1],
                    [-R2, Z, R0],
                    [R1, -R0, Z],
                ]
            )
        self._X = X

    def f_m(self, m):
        return self._X @ (self.mapping * m)

    def f_m_deriv(self, m):
        return self._X @ self.mapping.deriv(m)

    @property
    def W(self):
        if getattr(self, "_W", None) is None:
            mesh = self.regularization_mesh
            nC = mesh.nC

            weights = np.ones(
                nC,
            )
            for value in self._weights.values():
                if value.shape == (nC,):
                    weights *= value
                elif value.size == (mesh.dim * nC,):
                    weights *= np.linalg.norm(
                        value.reshape((nC, mesh.dim), order="F"), axis=1
                    )
            weights = np.sqrt(weights)
            self._W = sp.diags(np.r_[weights, weights, weights], format="csr")
        return self._W
