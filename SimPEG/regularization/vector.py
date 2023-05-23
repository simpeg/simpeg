from __future__ import annotations

# Regularizations for vector models.

import scipy.sparse as sp
import numpy as np
from .base import BaseRegularization, Smallness


class BaseVectorRegularization(BaseRegularization):
    """Base regularizer for models where each value is a vector.

    Used when your model has a multiple parameters for each cell. This can be helpful if
    your model is made up of vector values in each cell or it is an anisotropic model.
    """

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        mesh = self.regularization_mesh
        return [(mesh.nC,), (mesh.dim * mesh.nC,), (mesh.nC, mesh.dim)]


class CrossReferenceRegularization(Smallness, BaseVectorRegularization):
    """Vector regularization with a reference direction.

    This regularizer measures the magnitude of the cross product of the vector model
    with a reference vector model. This encourages the vectors in the model to point
    in the reference direction. The cross product of two vectors is minimized when they
    are parallel (or anti-parallel) to each other, and maximized when the vectors are
    perpendicular to each other.

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
    mapping : SimPEG.maps.IdentityMap, optional
        An optional linear mapping that would go from the model space to the space where
        the cross-product is enforced.
    weights : dict of [str: array_like], optional
        Any cell based weights for the regularization. Note if given a weight that is
        (n_cells, dim), meaning it is dependent on the vector component, it will compute
        the geometric mean of the component weights per cell and use that as a weight.
    **kwargs
        Arguments passed on to the parent classes: :py:class`.Smallness` and
        :py:class`.BaseVectorRegularization`.

    Notes
    -----
    The continuous form of this regularization looks like:

    .. math::
        \phi_{cross}(m) = \int_{V} ||\vec{m} \times \vec{m}_{ref}||^2 dV
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
        """The reference direction model

        Returns
        -------
        (n_active, dim) numpy.ndarray
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
            if mesh.dim == 2:
                diag = weights
            else:
                diag = np.r_[weights, weights, weights]
            self._W = sp.diags(diag, format="csr")
        return self._W
