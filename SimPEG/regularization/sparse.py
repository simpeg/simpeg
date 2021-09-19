from __future__ import annotations

import numpy as np

from .base import BaseRegularization, BaseComboRegularization, Small, SmoothDeriv
from .. import utils


class BaseSparse(BaseRegularization):
    """
    Base class for building up the components of the Sparse Regularization
    """
    _irls_scaled = True
    _irls_threshold = 1e-8
    _norm = 2.

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

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
        if self._irls_threshold is None:
            raise AttributeError("'irls_threhsold' must be set on the regularization.")
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        if value <= 0:
            raise ValueError("Value of 'irls_threshold' should be larger than 0.")

        self._irls_threshold = value

    @property
    def norm(self):
        """
        Value of the norm
        """
        return self._norm

    @norm.setter
    def norm(self, value):
        if (value < 0) or (value > 2):
            raise ValueError(
                "Value provided for 'norm' should be in the interval [0, 2]"
            )
        self._norm = value


class SparseSmall(BaseSparse, Small):
    """
    Sparse smallness regularization

    **Inputs**

    :param int norm: norm on the smallness
    """

    _multiplier_pair = "alpha_s"

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh=mesh, **kwargs)

    @property
    def W(self):
        """
        Weighting matrix
        """
        weights = self.free_multiplier * self.regularization_mesh.vol

        if self.cell_weights is not None:
            weights *= self.cell_weights

        if self.free_weights is not None:
            weights *= self.free_weights

        weights *= self.free_weights
        return utils.sdiag(weights ** 0.5)

    @property
    def free_weights(self):
        if getattr(self, "_free_weights", None) is None:
            if self.model is None or self.norm == 2:
                self._free_weights = np.ones(self.mapping.shape[0])
            else:
                self.irls_weights(self.model)
        return self._free_weights

    @free_weights.setter
    def free_weights(self, value):
        self._free_weights = value

    def irls_weights(self, m):
        """
        Compute and store the irls weights.
        """
        f_m = self.f_m(m)
        eta = np.ones_like(f_m)

        if self.irls_scaled:
            # Scale on l2-norm gradient: f_m.max()
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()

            # Compute theoritical maximum gradients for p < 1
            maxVal[self.norm < 1] = self.irls_threshold / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            maxGrad = maxVal / (maxVal ** 2.0 + self.irls_threshold ** 2.0) ** (
                    1.0 - self.norm / 2.0
            )
            eta[maxGrad != 0] = np.abs(f_m).max() / maxGrad[maxGrad != 0]

        # Scaled IRLS weights
        r = (eta / (f_m ** 2.0 + self.irls_threshold ** 2.0) ** (1.0 - self.norm / 2.0)) ** 0.5
        self.free_weights = r  # stash on the first calculation


class SparseDeriv(BaseSparse, SmoothDeriv):
    """
    Base Class for sparse regularization on first spatial derivatives
    """
    _gradient_type = "total"

    def __init__(self, mesh, orientation="x", **kwargs):
        self.orientation = orientation
        super().__init__(mesh=mesh, **kwargs)

    @property
    def W(self):
        """
        Weighting matrix that takes the volumes, free weights, fixed weights and
        length scales of the difference operator (normalized optional).
        """
        average_cell_face = getattr(self.regularization_mesh, "aveCC2F{}".format(self.orientation))
        weights = self.free_multiplier * self.regularization_mesh.vol

        if self.cell_weights is not None:
            weights *= self.cell_weights

        return utils.sdiag(
            self.length_scales *
            ((average_cell_face * weights) * self.free_weights) ** 0.5
        )

    @property
    def free_weights(self):
        if getattr(self, "_free_weights", None) is None:
            if self.model is None or self.norm == 2:
                self._free_weights = np.ones_like(self.length_scales)
            else:
                self.irls_weights(self.model)
        return self._free_weights

    @free_weights.setter
    def free_weights(self, value):
        self._free_weights = value

    def irls_weights(self, m):
        """
        Compute and store the irls weights.
        """
        if self.gradient_type == "total":
            if self.reference_model_in_smooth:
                delta_m = self.mapping * self._delta_m(m)
            else:
                delta_m = self.mapping * m

            f_m = np.zeros_like(delta_m)
            for ii, comp in enumerate("xyz"):
                if self.regularization_mesh.dim > ii:
                    dm_dl = getattr(self.regularization_mesh, f"cellDiff{comp}Stencil") * delta_m

                    if self.model_units == "radian":
                        dm_dl = utils.mat_utils.coterminal(dm_dl)

                    f_m += np.abs(
                        getattr(self.regularization_mesh, f"aveF{comp}2CC") *
                        dm_dl
                    )
            f_m = getattr(self.regmesh, f"aveCC2F{self.orientation}") * f_m
        else:
            f_m = self.f_m(m)

        eta = np.ones_like(f_m)
        if self.irls_scaled:
            # Scale on l2-norm gradient: f_m.max()
            maxVal = np.ones_like(f_m) * np.abs(f_m).max()

            # Compute theoritical maximum gradients for p < 1
            maxVal[self.norm < 1] = self.irls_threshold / np.sqrt(
                1.0 - self.norm[self.norm < 1]
            )
            maxGrad = maxVal / (maxVal ** 2.0 + self.irls_threshold ** 2.0) ** (
                    1.0 - self.norm / 2.0
            )
            eta[maxGrad != 0] = np.abs(f_m).max() / maxGrad[maxGrad != 0]

        # Scaled IRLS weights
        r = (eta / (f_m ** 2.0 + self.irls_threshold ** 2.0) ** (1.0 - self.norm / 2.0)) ** 0.5
        self.free_weights = r  # stash on the first calculation

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


class Sparse(BaseComboRegularization):
    """
    The regularization is:

    .. math::

        R(m) = \\frac{1}{2}\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R
        W(m-m_\\text{ref})}

    where the IRLS weight

    .. math::

        R = \eta TO FINISH LATER!!!

    So the derivative is straight forward:

    .. math::

        R(m) = \mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

    The IRLS weights are recomputed after each beta solves.
    It is strongly recommended to do a few Gauss-Newton iterations
    before updating.
    """
    _irls_scaled = True
    _irls_threshold = 1e-8
    _gradient_type = "total"
    _norms = [2, 2, 2, 2]

    def __init__(
        self, mesh, alpha_s=1.0, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0, **kwargs
    ):

        objfcts = [
            SparseSmall(mesh=mesh),
            SparseDeriv(mesh=mesh, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(SparseDeriv(mesh=mesh, orientation="y"))

        if mesh.dim > 2:
            objfcts.append(SparseDeriv(mesh=mesh, orientation="z"))

        super().__init__(
            mesh=mesh,
            objfcts=objfcts,
            alpha_s=alpha_s,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            alpha_z=alpha_z,
            **kwargs
        )

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

        for fct in self.objfcts:
            if hasattr(fct, "_gradient_type"):
                fct._gradient_type = value

    @property
    def norms(self):
        """
        Value of the norm
        """
        return self._norms

    @norms.setter
    def norms(self, values: list | np.ndarray | float):
        if isinstance(values, list) or isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=float).flatten()

            if len(values) != len(self.objfcts):
                raise ValueError(
                    "The number of values provided for 'norms' does not "
                    "match the number of regularization functions."
                )


        elif isinstance(values, int) or isinstance(values, float):
            values = [float(values)] * len(self.objfcts)
        else:
            raise TypeError("Input 'norms' must be a float, list or array of values")

        if np.any(values < 0) or np.any(values > 2):
            raise ValueError(
                "Value provided for 'norms' should be in the interval [0, 2]"
            )

        self._norms = values

        for val, fct in zip(values, self.objfcts):
            fct.norm = val

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

        for fct in self.objfcts:
            fct.irls_scaled = value

    @property
    def irls_threshold(self):
        """
        Constant added to the denominator of the IRLS weights for stability.
        """
        return self._irls_threshold

    @irls_threshold.setter
    def irls_threshold(self, value):
        if value <= 0:
            raise ValueError("Value of 'irls_threshold' should be larger than 0.")

        self._irls_threshold = value

        for fct in self.objfcts:
            fct.irls_threshold = value