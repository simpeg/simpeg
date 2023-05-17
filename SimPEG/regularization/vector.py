from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.sparse as sp
import numpy as np
from .base import Smallness
from discretize.base import BaseMesh
from .base import RegularizationMesh, BaseRegularization
from .sparse import Sparse, SparseSmallness, SparseSmoothness

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

# Regularizations for vector models.


class BaseVectorRegularization(BaseRegularization):
    """The regularizers work on models where each value is a vector."""

    @property
    def n_comp(self):
        """Number of components in the model."""
        return int(self.mapping.shape[0] / self.regularization_mesh.nC)

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        mesh = self.regularization_mesh
        return [(mesh.nC,), (self.n_comp * mesh.nC,), (mesh.nC, self.n_comp)]


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
        if value.shape != (nC, self.n_comp):
            if value.shape == (self.n_comp,):
                # expand it out for each mesh cell
                value = np.tile(value, (nC, 1))
            else:
                raise ValueError(f"ref_dir must be shape {(nC, self.n_comp)}")
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
                elif value.size == (self.n_comp * nC,):
                    weights *= np.linalg.norm(
                        value.reshape((nC, self.n_comp), order="F"), axis=1
                    )
            weights = np.sqrt(weights)
            self._W = sp.diags(np.r_[weights, weights, weights], format="csr")
        return self._W


class BaseAmplitude(BaseVectorRegularization):
    """
    Base vector amplitude class.
    Requires a mesh and a :obj:`SimPEG.maps.Wires` mapping.
    """

    def amplitude(self, m):
        return np.linalg.norm(
            (self.mapping * self._delta_m(m)).reshape(
                (self.regularization_mesh.nC, self.n_comp), order="F"
            ),
            axis=1,
        )

    def deriv(self, m) -> np.ndarray:
        """ """
        d_m = self._delta_m(m)

        return self.f_m_deriv(m).T * (
            self.W.T @ self.W @ (self.f_m_deriv(m) @ d_m).reshape((-1, self.n_comp), order='F')
        ).flatten(order='F')

    def deriv2(self, m, v=None) -> csr_matrix:
        """ """
        f_m_deriv = self.f_m_deriv(m)

        if v is None:
            return f_m_deriv.T * (
                sp.block_diag([self.W.T * self.W] * self.n_comp) * f_m_deriv
            )

        return f_m_deriv.T * (
            self.W.T @ self.W @ (f_m_deriv * v).reshape((-1, self.n_comp), order='F')
        ).flatten(order='F')


class AmplitudeSmallness(SparseSmallness, BaseAmplitude):
    """
    Sparse smallness regularization on vector amplitude.
    """

    def f_m(self, m):
        """
        Compute the amplitude of a vector model.
        """

        return self.amplitude(m)

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
                elif value.shape == (self.n_comp * nC,):
                    weights *= np.linalg.norm(
                        value.reshape((nC, self.n_comp), order="F"), axis=1
                    )

            self._W = sp.diags(np.sqrt(weights), format="csr")

        return self._W


class AmplitudeSmoothnessFirstOrder(SparseSmoothness, BaseAmplitude):
    """
    Sparse first spatial derivatives of amplitude.
    """

    @property
    def _weights_shapes(self) -> list[tuple[int]]:
        """Acceptable lengths for the weights

        Returns
        -------
        list of tuple
            Each tuple represents accetable shapes for the weights
        """
        nC = self.regularization_mesh.nC
        nF = getattr(
            self.regularization_mesh, "aveCC2F{}".format(self.orientation)
        ).shape[0]
        return [
            (nF,),
            (self.n_comp * nF,),
            (nF, self.n_comp),
            (nC,),
            (self.n_comp * nC,),
            (nC, self.n_comp),
        ]

    def f_m(self, m):
        a = self.amplitude(m)

        return self.cell_gradient @ a

    def f_m_deriv(self, m) -> csr_matrix:
        """"""
        return sp.block_diag([self.cell_gradient] * self.n_comp) @ self.mapping.deriv(
            self._delta_m(m)
        )

    @property
    def W(self):
        """
        Weighting matrix that takes the volumes, free weights, fixed weights and
        length scales of the difference operator (normalized optional).
        """
        if getattr(self, "_W", None) is None:
            average_cell_2_face = getattr(
                self.regularization_mesh, "aveCC2F{}".format(self.orientation)
            )
            nC = self.regularization_mesh.nC
            nF = average_cell_2_face.shape[0]
            weights = 1.0
            for values in self._weights.values():
                if values.shape[0] == nC:
                    values = average_cell_2_face * values
                elif not values.shape == (nF,):
                    values = np.linalg.norm(
                        values.reshape((-1, self.n_comp), order="F"), axis=1
                    )
                    if values.size == nC:
                        values = average_cell_2_face * values

                weights *= values

            self._W = sp.diags(np.sqrt(weights), format="csr")

        return self._W


class VectorAmplitude(Sparse):
    r"""
    The regularization is:

    The function defined here approximates:

    .. math::

        \phi_m(\mathbf{m}) = \alpha_s \| \mathbf{W}_s \; \mathbf{a}(\mathbf{m} - \mathbf{m_{ref}) \|_p
        + \alpha_x \| \mathbf{W}_x \; \frac{\partial}{\partial x} \mathbf{a}(\mathbf{m} - \mathbf{m_{ref}) \|_p
        + \alpha_y \| \mathbf{W}_y \; \frac{\partial}{\partial y} \mathbf{a}(\mathbf{m} - \mathbf{m_{ref}) \|_p
        + \alpha_z \| \mathbf{W}_z \; \frac{\partial}{\partial z} \mathbf{a}(\mathbf{m} - \mathbf{m_{ref}) \|_p

    where $\mathbf{a}(\mathbf{m} - \mathbf{m_{ref})$ is the vector amplitude of the difference between
    the model and the reference model.

    .. math::

        \mathbf{a}(\mathbf{m} - \mathbf{m_{ref}) = [\sum_{i}^{N}(\mathbf{P}_i\;(\mathbf{m} - \mathbf{m_{ref}}))^{2}]^{1/2}

    where :math:`\mathbf{P}_i` is the projection of i-th component of the vector model with N-dimensions.

    """

    def __init__(
        self,
        mesh,
        mapping=None,
        active_cells=None,
        **kwargs,
    ):
        if not isinstance(mesh, (BaseMesh, RegularizationMesh)):
            raise TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )

        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        self._regularization_mesh = mesh

        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        objfcts = [
            AmplitudeSmallness(mesh=self.regularization_mesh, mapping=mapping),
            AmplitudeSmoothnessFirstOrder(
                mesh=self.regularization_mesh, orientation="x", mapping=mapping
            ),
        ]

        if mesh.dim > 1:
            objfcts.append(
                AmplitudeSmoothnessFirstOrder(
                    mesh=self.regularization_mesh, orientation="y", mapping=mapping
                )
            )

        if mesh.dim > 2:
            objfcts.append(
                AmplitudeSmoothnessFirstOrder(
                    mesh=self.regularization_mesh, orientation="z", mapping=mapping
                )
            )

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            mapping=mapping,
            **kwargs,
        )
