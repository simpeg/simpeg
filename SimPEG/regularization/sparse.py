from __future__ import annotations

import numpy as np

from discretize.base import BaseMesh

from .base import (
    BaseRegularization,
    WeightedLeastSquares,
    RegularizationMesh,
    Smallness,
    SmoothnessFirstOrder,
)
from .. import utils


class BaseSparse(BaseRegularization):
    """
    Base class for building up the components of the Sparse Regularization
    """

    def __init__(self, mesh, norm=2.0, irls_scaled=True, irls_threshold=1e-8, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.norm = norm
        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def irls_scaled(self) -> bool:
        """
        Scale irls weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                "'irls_scaled must be of type 'bool'. "
                f"Value of type {type(value)} provided."
            )
        self._irls_scaled = value

    @property
    def irls_threshold(self):
        """
        Constant added to the denominator of the IRLS weights for stability.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("Value of 'irls_threshold' should be greater than 0.")
        self._irls_threshold = value

    @property
    def norm(self):
        """
        Value of the norm
        """
        return self._norm

    @norm.setter
    def norm(self, value: float | np.ndarray | None):
        if value is None:
            value = np.ones(self._weights_shapes[0]) * 2.0
        else:
            if isinstance(value, (float, int)):
                value = np.ones(self._weights_shapes[0]) * value

            if np.any(value < 0) or np.any(value > 2):
                raise ValueError(
                    "Value provided for 'norm' should be in the interval [0, 2]"
                )
        self._norm = value

    def get_lp_weights(self, f_m):
        """
        Utility function to get the IRLS weights.
        By default, the weights are scaled by the gradient of the IRLS on
        the max of the l2-norm.
        """
        lp_scale = np.ones_like(f_m)
        if self.irls_scaled:
            # Scale on l2-norm gradient: f_m.max()
            l2_max = np.ones_like(f_m) * np.abs(f_m).max()
            # Compute theoretical maximum gradients for p < 1
            l2_max[self.norm < 1] = self.irls_threshold / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            lp_values = l2_max / (l2_max ** 2.0 + self.irls_threshold ** 2.0) ** (
                1.0 - self.norm / 2.0
            )
            lp_scale[lp_values != 0] = np.abs(f_m).max() / lp_values[lp_values != 0]

        return lp_scale / (f_m ** 2.0 + self.irls_threshold ** 2.0) ** (
            1.0 - self.norm / 2.0
        )


class SparseSmallness(BaseSparse, Smallness):
    """
    Sparse smallness regularization

    **Inputs**

    :param int norm: norm on the smallness
    """

    _multiplier_pair = "alpha_s"

    def update_weights(self, m):
        """
        Compute and store the irls weights.
        """
        f_m = self.f_m(m)
        self.set_weights(irls=self.get_lp_weights(f_m))


class SparseSmoothness(BaseSparse, SmoothnessFirstOrder):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def __init__(self, mesh, orientation="x", gradient_type="total", **kwargs):
        if "gradientType" in kwargs:
            self.gradientType = kwargs.pop("gradientType")
        else:
            self.gradient_type = gradient_type
        super().__init__(mesh=mesh, orientation=orientation, **kwargs)

    def update_weights(self, m):
        """
        Compute and store the irls weights.
        """
        if self.gradient_type == "total":

            delta_m = self.mapping * self._delta_m(m)
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

    @property
    def gradient_type(self) -> str:
        """
        Choice of gradient measure used in the irls weights
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        if value not in ["total", "components"]:
            raise TypeError(
                "Value for 'gradient_type' must be 'total' or 'components'. "
                f"Value {value} provided."
            )
        self._gradient_type = value

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.x.0", error=False, future_warn=False
    )


class Sparse(WeightedLeastSquares):
    """
    The regularization is:

    .. math::

        R(m) = \\frac{1}{2}\\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R
        W(m-m_\\text{ref})}

    where the IRLS weight

    .. math::

        R = \\eta \\text{diag} \\left[\\mathbf{r}_s \\right]^{1/2} \\
        r_{s_i} = {\\Big( {({m_i}^{(k-1)})}^{2} + \\epsilon^2 \\Big)}^{p_s/2 - 1}

    where k denotes the iteration number. So the derivative is straight forward:

    .. math::

        R(m) = \\mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

    The IRLS weights are re-computed after each beta solves using
    :obj:`~SimPEG.directives.Update_IRLS` within the inversion directives.
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        norms=None,
        gradient_type="total",
        irls_scaled=True,
        irls_threshold=1e-8,
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
            SparseSmallness(mesh=self.regularization_mesh),
            SparseSmoothness(mesh=self.regularization_mesh, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(SparseSmoothness(mesh=self.regularization_mesh, orientation="y"))

        if mesh.dim > 2:
            objfcts.append(SparseSmoothness(mesh=self.regularization_mesh, orientation="z"))

        gradientType = kwargs.pop("gradientType", None)
        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )
        if norms is None:
            norms = [1] * (mesh.dim + 1)
        self.norms = norms

        if gradientType is not None:
            # Trigger deprecation warning
            self.gradientType = gradientType
        else:
            self.gradient_type = gradient_type

        self.irls_scaled = irls_scaled
        self.irls_threshold = irls_threshold

    @property
    def gradient_type(self) -> str:
        """
        Choice of gradient measure used in the irls weights
        """
        return self._gradient_type

    @gradient_type.setter
    def gradient_type(self, value: str):
        for fct in self.objfcts:
            if hasattr(fct, "gradient_type"):
                fct.gradient_type = value

        self._gradient_type = value

    gradientType = utils.code_utils.deprecate_property(
        gradient_type, "gradientType", "0.x.0", error=False, future_warn=False
    )

    @property
    def norms(self):
        """
        Value of the norm
        """
        return self._norms

    @norms.setter
    def norms(self, values: list | np.ndarray | None):
        if values is not None:
            if len(values) != len(self.objfcts):
                raise ValueError(
                    "The number of values provided for 'norms' does not "
                    "match the number of regularization functions."
                )
        else:
            values = [None] * len(self.objfcts)

        for val, fct in zip(values, self.objfcts):
            fct.norm = val

        self._norms = values

    @property
    def irls_scaled(self) -> bool:
        """
        Scale irls weights.
        """
        return self._irls_scaled

    @irls_scaled.setter
    def irls_scaled(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                "'irls_scaled must be of type 'bool'. "
                f"Value of type {type(value)} provided."
            )
        for fct in self.objfcts:
            fct.irls_scaled = value
        self._irls_scaled = value

    @property
    def irls_threshold(self):
        """
        Constant added to the denominator of the IRLS weights for stability.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("Value of 'irls_threshold' should be greater than 0.")

        self._irls_threshold = value

        for fct in self.objfcts:
            fct.irls_threshold = value

    def update_weights(self, model):
        """
        Trigger irls update on all children
        """
        for fct in self.objfcts:
            fct.update_weights(model)
