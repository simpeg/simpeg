from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import TYPE_CHECKING

from discretize.base import BaseMesh
from SimPEG import maps
from .base import (
    RegularizationMesh,
    BaseRegularization
)
from .sparse import (
    Sparse,
    SparseSmall,
    SparseDeriv
)
from .. import utils

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseVectorAmplitude(BaseRegularization):
    """
    Base vector amplitude function.
    """
    _projection = None
    _W = None

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

    @property
    def mapping(self) -> maps.Wires:
        return self._mapping

    @mapping.setter
    def mapping(self, wires):
        if not isinstance(wires, maps.Wires):
            raise ValueError(f"A 'mapping' of type {maps.Wires} must be provided.")

        for wire in wires.maps:
            if wire[1].shape[0] != self.regularization_mesh.nC:
                raise ValueError(
                    f"All models must be the same size! Got {wire} with shape{wire[1].shape[0]}"
                )
        self._mapping = wires

    def amplitude_map(self, m):
        """Create sparse vector model."""
        # amplitude = 0
        # for _, wire in self.mapping.maps:
        #     amplitude += (wire * m)**2.

        return np.linalg.norm(m, axis=0)

    def set_weights(self, **weights):
        """Adds (or updates) the specified weights to the regularization

        Parameters:
        -----------
        **kwargs : key, numpy.ndarray
            Each keyword argument is added to the weights used by the regularization.
            They can be accessed with their keyword argument.

        Examples
        --------
        >>> import discretize
        >>> from SimPEG.regularization import Small
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Small(mesh)
        >>> reg.set_weights(my_weight=np.ones(mesh.n_cells))
        >>> reg.get_weights('my_weight')
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        """
        for key, values in weights.items():
            if not isinstance(values, tuple):
                values = (values,) * len(self.mapping.maps)

            if len(values) != len(self.mapping.maps):
                raise ValueError(f"Values provided for weight {key} must be of tuple of len({len(self.mapping.maps)})")

            self._weights[key] = {}
            for (name, _), value in zip(self.mapping.maps, values):
                self.validate_array_type("weights", value, float)
                self.validate_shape("weights", value, self._weights_shapes)
                self._weights[key][name] = value

        self._W = None

    @utils.timeIt
    def __call__(self, m):
        """
        """
        r = self.W * self.f_m(m)
        return 0.5 * r.dot(r)

    @utils.timeIt
    def deriv(self, m) -> np.ndarray:
        """
        """
        r = self.W * self.f_m(m)
        return self.f_m_deriv(m).T * (self.W.T * r)

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        """
        """
        f_m_deriv = self.f_m_deriv(m)
        if v is None:
            return f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)

        return f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))


class VectorAmplitudeSmall(SparseSmall, BaseVectorAmplitude):
    """
    Sparse smallness regularization on vector amplitude.

    **Inputs**

    :param int norm: norm on the smallness
    """

    def f_m(self, m):
        """
        Compute the amplitude of a vector model.
        """

        return self.amplitude_map(self.mapping * self._delta_m(m))

    def f_m_deriv(self, m) -> csr_matrix:

        return self.mapping.deriv(self._delta_m(m))

    @property
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            self._W = []

            for name, _ in self.mapping.maps:
                self._W.append(1.0)
                for weight in self._weights.values():
                    self._W[-1] *= weight[name]

                self._W[-1] = utils.sdiag(self._W[-1] ** 0.5)

            self._W = sp.block_diag(self._W)
        return self._W


class VectorAmplitudeDeriv(SparseDeriv, BaseVectorAmplitude):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def f_m(self, m):
        m = self.amplitude_map(self.mapping * self._delta_m(m))
        dfm_dl = self.cell_gradient @ m

        return dfm_dl

    def f_m_deriv(self, m) -> csr_matrix:
        m = self.amplitude_map(self.mapping * self._delta_m(m))

        deriv = []
        for _, wire in self.mapping.maps:
            deriv += [self.cell_gradient @ wire.deriv(m)]
        return sp.vstack(deriv)

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
            self._W = []

            for name, _ in self.mapping.maps:

                self._W.append(-1)

                for weight in self._weights.values():
                    values = weight[name]
                    if values.shape[0] == self.regularization_mesh.nC:
                        values = average_cell_2_face * values

                    self._W[-1] *= values

                self._W[-1] = utils.sdiag(self._W[-1] ** 0.5)

            self._W = sp.block_diag(self._W)

        return self._W


class VectorAmplitude(Sparse):
    """
    The regularization is:
    ...
    """

    def __init__(
        self,
        mesh,
        wire_map,
        active_cells=None,
        **kwargs,
    ):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh

        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        objfcts = [
            VectorAmplitudeSmall(mesh=self.regularization_mesh, mapping=wire_map),
            VectorAmplitudeDeriv(mesh=self.regularization_mesh, mapping=wire_map, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(VectorAmplitudeDeriv(mesh=self.regularization_mesh, mapping=wire_map, orientation="y"))

        if mesh.dim > 2:
            objfcts.append(VectorAmplitudeDeriv(mesh=self.regularization_mesh, mapping=wire_map, orientation="z"))

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            mapping=wire_map,
            **kwargs,
        )

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, wires):
        if not isinstance(wires, maps.Wires):
            raise ValueError(f"A 'mapping' of type {maps.Wires} must be provided.")

        for wire in wires.maps:
            if wire[1].shape[0] != self.regularization_mesh.nC:
                raise ValueError(
                    f"All models must be the same size! Got {wire} with shape{wire[1].shape[0]}"
                )
        self._mapping = wires

        for fct in self.objfcts:
            fct.mapping = wires

