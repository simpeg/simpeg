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

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

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

    def amplitude_map(self, m):
        """Create sparse vector model."""
        amplitude = 0
        for wire_model in (self.mapping * m):
            amplitude += wire_model**2.

        return amplitude**0.5


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
        return self.cell_gradient @ self.mapping.deriv(m)


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

