from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from discretize.base import BaseMesh
from SimPEG import maps
from .base import RegularizationMesh, BaseRegularization
from .sparse import Sparse, SparseSmallness, SparseSmoothness
from .. import utils
from SimPEG.utils.code_utils import validate_ndarray_with_shape

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseAmplitude(BaseRegularization):
    """
    Base vector amplitude class.
    Requires a mesh and a :obj:`SimPEG.maps.Wires` mapping.
    """

    _W = None

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

    @property
    def mapping(self) -> maps.Wires:
        return self._mapping

    @mapping.setter
    def mapping(self, wires: maps.Wires | None):
        if isinstance(wires, type(None)):
            wires = maps.Wires(("model", self.regularization_mesh.nC))

        elif not isinstance(wires, maps.Wires):
            raise TypeError(f"A 'mapping' of type {maps.Wires} must be provided.")

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
            if isinstance(values, tuple):
                if len(values) != len(self.mapping.maps):
                    raise ValueError(
                        f"Values provided for weight {key} must be of tuple of len({len(self.mapping.maps)})"
                    )

                for value in values:
                    validate_ndarray_with_shape(
                        "weights", value, shape=self._weights_shapes, dtype=float
                    )

                self._weights[key] = np.linalg.norm(np.vstack(values), axis=0)
            else:
                validate_ndarray_with_shape(
                    "weights", values, shape=self._weights_shapes, dtype=float
                )
                self._weights[key] = values

        self._W = None

    @utils.timeIt
    def deriv(self, m) -> np.ndarray:
        """ """
        d_m = self._delta_m(m)

        deriv = 0.0

        for f_m_deriv in self.f_m_deriv(d_m):
            deriv += f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv * d_m)

        return deriv

    @utils.timeIt
    def deriv2(self, m, v=None) -> csr_matrix:
        """ """
        deriv = 0.0

        for f_m_deriv in self.f_m_deriv(m):
            if v is None:
                deriv += f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv)
            else:
                deriv += f_m_deriv.T * ((self.W.T * self.W) * f_m_deriv * v)

        return deriv

    @property
    def _nC_residual(self) -> int:
        """
        Shape of the residual
        """
        if self.mapping is None:
            raise AttributeError("The regularization does not have a 'mapping' yet.")

        return int(np.sum([wire.shape[0] for (_, wire) in self.mapping.maps]))


class AmplitudeSmallness(SparseSmallness, BaseAmplitude):
    """
    Sparse smallness regularization on vector amplitude.
    """

    def f_m(self, m):
        """
        Compute the amplitude of a vector model.
        """

        return np.linalg.norm(self.mapping * self._delta_m(m), axis=0)

    def f_m_deriv(self, m) -> csr_matrix:
        deriv = []
        dm = self._delta_m(m)
        for _, wire in self.mapping.maps:
            deriv += [wire.deriv(dm)]
        return deriv


class AmplitudeSmoothnessFirstOrder(SparseSmoothness, BaseAmplitude):
    """
    Sparse first spatial derivatives of amplitude.
    """

    def f_m(self, m):
        a = np.linalg.norm(self.mapping * self._delta_m(m), axis=0)

        return self.cell_gradient @ a

    def f_m_deriv(self, m) -> csr_matrix:
        deriv = []
        dm = self._delta_m(m)
        for _, wire in self.mapping.maps:
            deriv += [self.cell_gradient * wire.deriv(dm)]

        return deriv

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
                    f_m += np.abs(
                        getattr(self.regularization_mesh, f"aveF{comp}2CC") * dm
                    )

            f_m = getattr(self.regularization_mesh, f"aveCC2F{self.orientation}") * f_m

        else:
            f_m = self.f_m(m)

        self.set_weights(irls=self.get_lp_weights(f_m))


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
    ...
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

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, wires):
        if not isinstance(wires, maps.Wires):
            raise TypeError(f"A 'mapping' of type {maps.Wires} must be provided.")

        for wire in wires.maps:
            if wire[1].shape[0] != self.regularization_mesh.nC:
                raise ValueError(
                    f"All models must be the same size! Got {wire} with shape{wire[1].shape[0]}"
                )
        self._mapping = wires

        for fct in self.objfcts:
            fct.mapping = wires
