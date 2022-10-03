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
    SparseSmallness,
    SparseSmoothnessFirstOrder
)
from .. import utils
from SimPEG.utils.code_utils import validate_array_type, validate_shape

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseAmplitude(BaseRegularization):
    """
    Base vector amplitude function.
    """
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
        >>> from SimPEG.regularization import Smallness
        >>> mesh = discretize.TensorMesh([2, 3, 2])
        >>> reg = Smallness(mesh)
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
                validate_array_type("weights", value, float)
                validate_shape("weights", value, self._weights_shapes)
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
        f_m_derivs = 0.
        for f_m_deriv in self.f_m_deriv(m):
            f_m_derivs += f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv * m)
        return f_m_derivs

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        """
        """
        f_m_derivs = 0.
        for f_m_deriv in self.f_m_deriv(m):
            if v is None:
                f_m_derivs += f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)
            else:
                f_m_derivs += f_m_deriv.T * (self.W.T * (self.W * (f_m_deriv * v)))

        return f_m_derivs


class AmplitudeSmallness(SparseSmallness, BaseAmplitude):
    """
    Sparse smallness regularization on vector amplitude.

    **Inputs**

    :param int norm: norm on the smallness
    """

    def f_m(self, m):
        """
        Compute the amplitude of a vector model.
        """

        return np.linalg.norm(self.mapping * self._delta_m(m), axis=0)

    def f_m_deriv(self, m) -> csr_matrix:

        return self.mapping.deriv(self._delta_m(m))

    @property
    def W(self):
        """
        Weighting matrix
        """
        if getattr(self, "_W", None) is None:
            weights = []

            for name, _ in self.mapping.maps:
                weights.append(1.0)
                for weight in self._weights.values():
                    weights[-1] *= weight[name]

                weights[-1] = utils.sdiag(weights[-1] ** 0.5)

            self._W = sp.vstack(weights)
        return self._W


class AmplitudeSmoothnessFirstOrder(SparseSmoothnessFirstOrder, BaseAmplitude):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def f_m(self, m):
        a = np.linalg.norm(self.mapping * self._delta_m(m), axis=0)

        return self.cell_gradient @ a

    def f_m_deriv(self, m) -> csr_matrix:

        deriv = []
        for map_deriv in self.mapping.deriv(self._delta_m(m)):
            deriv.append(self.cell_gradient * map_deriv)

        return deriv

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
            weights = []

            for name, _ in self.mapping.maps:

                weights.append(1.0)

                for weight in self._weights.values():
                    values = weight[name]
                    if values.shape[0] == self.regularization_mesh.nC:
                        values = average_cell_2_face * values

                    weights[-1] *= values

                weights[-1] = utils.sdiag(weights[-1] ** 0.5)

            self._W = sp.vstack(weights)

        return self._W

    def update_weights(self, m):
        """
        Compute and store the irls weights.
        """
        if self.gradient_type == "total":
            delta_m = self.mapping * self._delta_m(m)
            delta_m = np.linalg.norm(delta_m, axis=0)
            f_m = np.zeros_like(delta_m)

            for ii, comp in enumerate("xyz"):
                if self.regularization_mesh.dim > ii:
                    dm = (
                        getattr(self.regularization_mesh, f"cell_gradient_{comp}")
                        * delta_m
                    )

                    if self.units is not None and self.units.lower() == "radian":
                        Ave = getattr(self.regularization_mesh, f"aveCC2F{comp}")
                        length_scales = Ave * (
                                self.regularization_mesh.Pac.T
                                * self.regularization_mesh.mesh.h_gridded[:, ii]
                        )
                        dm = utils.mat_utils.coterminal(dm * length_scales) / length_scales

                    f_m += np.abs(
                        getattr(self.regularization_mesh, f"aveF{comp}2CC") * dm
                    )

            f_m = getattr(self.regularization_mesh, f"aveCC2F{self.orientation}") * f_m

        else:
            f_m = self.f_m(m)

        self.set_weights(irls=self.get_lp_weights(f_m))


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
            AmplitudeSmallness(mesh=self.regularization_mesh, mapping=wire_map),
            AmplitudeSmoothnessFirstOrder(mesh=self.regularization_mesh, mapping=wire_map, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(AmplitudeSmoothnessFirstOrder(mesh=self.regularization_mesh, mapping=wire_map, orientation="y"))

        if mesh.dim > 2:
            objfcts.append(AmplitudeSmoothnessFirstOrder(mesh=self.regularization_mesh, mapping=wire_map, orientation="z"))

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

