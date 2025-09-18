from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass

from ..maps import Projection
from ._directives import InversionDirective, UpdatePreconditioner, BetaSchedule
from ..regularization import (
    Sparse,
    BaseSparse,
    SmoothnessFirstOrder,
    WeightedLeastSquares,
)
from ..utils import validate_integer, validate_float, deprecate_class


@dataclass
class IRLSMetrics:
    """
    Data class to store metrics used by the IRLS algorithm.

    Parameters
    ----------
    input_norms : list of floats or None
        List of norms temporarily stored during the initialization.
    irls_iteration_count : int
        Number of IRLS iterations.
    start_irls_iter : int or None
        Iteration number when the IRLS process started.
    f_old : float
        Previous value of the regularization function.
    """

    input_norms: list[float] | None = None
    irls_iteration_count: int = 0
    start_irls_iter: int | None = None
    f_old: float = 0.0


class UpdateIRLS(InversionDirective):
    """
    Directive to control the IRLS iterations for :class:`~simpeg.regularization.Sparse`.

    Parameters
    ----------
    cooling_rate: int
        Number of iterations to cool beta.
    cooling_factor: float
        Factor to cool beta.
    chifact_start: float
        Starting chi factor for the IRLS iterations.
    chifact_target: float
        Target chi factor for the IRLS iterations.
    irls_cooling_factor: float
        Factor to cool the IRLS threshold epsilon.
    f_min_change: float
        Minimum change in the regularization function to continue the IRLS iterations.
    max_irls_iterations: int
        Maximum number of IRLS iterations.
    misfit_tolerance: float
        Tolerance for the target misfit.
    percentile: float
        Percentile of the function values used to determine the initial IRLS threshold.
    verbose: bool
        Print information to the screen.
    """

    def __init__(
        self,
        cooling_rate: int = 1,
        cooling_factor: float = 2.0,
        chifact_start: float = 1.0,
        chifact_target: float = 1.0,
        irls_cooling_factor: float = 1.2,
        f_min_change: float = 1e-2,
        max_irls_iterations: int = 20,
        misfit_tolerance: float = 1e-1,
        percentile: float = 100.0,
        verbose: bool = True,
        **kwargs,
    ):
        self._metrics: IRLSMetrics | None = None
        self.cooling_rate = cooling_rate
        self.cooling_factor = cooling_factor
        self.chifact_start: float = chifact_start
        self.chifact_target: float = chifact_target
        self.irls_cooling_factor: float = irls_cooling_factor
        self.f_min_change: float = f_min_change
        self.max_irls_iterations: int = max_irls_iterations
        self.misfit_tolerance: float = misfit_tolerance
        self.percentile: float = percentile

        super().__init__(
            verbose=verbose,
            **kwargs,
        )

    @property
    def metrics(self) -> IRLSMetrics:
        """Various metrics used by the IRLS algorithm."""
        if self._metrics is None:
            self._metrics = IRLSMetrics()
        return self._metrics

    @property
    def max_irls_iterations(self) -> int:
        """Maximum irls iterations."""
        return self._max_irls_iterations

    @max_irls_iterations.setter
    def max_irls_iterations(self, value):
        self._max_irls_iterations = validate_integer(
            "max_irls_iterations", value, min_val=0
        )

    @property
    def misfit_tolerance(self) -> float:
        """Tolerance on deviation from the target chi factor, as a fractional percent."""
        return self._misfit_tolerance

    @misfit_tolerance.setter
    def misfit_tolerance(self, value):
        self._misfit_tolerance = validate_float("misfit_tolerance", value, min_val=0)

    @property
    def percentile(self) -> float:
        """Tolerance on deviation from the target chi factor, as a fractional percent."""
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        self._percentile = validate_float(
            "percentile", value, min_val=0.0, max_val=100.0
        )

    @property
    def chifact_start(self) -> float:
        """Target chi factor to start the IRLS process."""
        return self._chifact_start

    @chifact_start.setter
    def chifact_start(self, value):
        self._chifact_start = validate_float(
            "chifact_start", value, min_val=0, inclusive_min=False
        )

    @property
    def chifact_target(self) -> float:
        """Targer chi factor to maintain during the IRLS process."""
        return self._chifact_target

    @chifact_target.setter
    def chifact_target(self, value):
        self._chifact_target = validate_float(
            "chifact_target", value, min_val=0, inclusive_min=False
        )

    @property
    def cooling_factor(self):
        """Beta is divided by this value every :attr:`cooling_rate` iterations.

        Returns
        -------
        float
        """
        return self._cooling_factor

    @cooling_factor.setter
    def cooling_factor(self, value):
        self._cooling_factor = validate_float(
            "cooling_factor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def cooling_rate(self):
        """Cool beta after this number of iterations.

        Returns
        -------
        int
        """
        return self._cooling_rate

    @cooling_rate.setter
    def cooling_rate(self, value):
        self._cooling_rate = validate_integer("cooling_rate", value, min_val=1)

    @property
    def irls_cooling_factor(self) -> float:
        """IRLS threshold parameter (epsilon) is divided by this value every iteration."""
        return self._irls_cooling_factor

    @irls_cooling_factor.setter
    def irls_cooling_factor(self, value):
        self._irls_cooling_factor = validate_float(
            "irls_cooling_factor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def f_min_change(self) -> float:
        """Target chi factor to start the IRLS process."""
        return self._f_min_change

    @f_min_change.setter
    def f_min_change(self, value):
        self._f_min_change = validate_float(
            "f_min_change", value, min_val=0, inclusive_min=False
        )

    def misfit_from_chi_factor(self, chi_factor: float) -> float:
        """
        Compute the target misfit from the chi factor.

        Parameters
        ----------
        chi_factor : float
            Chi factor to compute the target misfit from.
        """
        value = 0

        for survey in self.survey:
            value += survey.nD * chi_factor

        return value

    def adjust_cooling_schedule(self):
        """
        Adjust the cooling schedule based on the misfit.
        """
        if self.metrics.start_irls_iter is not None:
            ratio = self.invProb.phi_d / self.misfit_from_chi_factor(
                self.chifact_target
            )
            if np.abs(1.0 - ratio) > self.misfit_tolerance:

                if ratio > 1:
                    update_ratio = 1 / np.mean([0.75, 1 / ratio])
                else:
                    update_ratio = 1 / np.mean([2.0, 1 / ratio])

                self.cooling_factor = update_ratio
            else:
                self.cooling_factor = 1.0

    def initialize(self):
        """
        Initialize the IRLS iterations with l2-norm regularization (mode:1).
        """

        input_norms = []
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                input_norms += [None]
            else:
                input_norms += [reg.norms]
                reg.norms = [2.0 for _ in reg.objfcts]

        self._metrics = IRLSMetrics(input_norms=input_norms)

    def endIter(self):
        """
        Check on progress of the inversion and start/update the IRLS process.
        """
        # Update the cooling factor (only after IRLS has started)
        self.adjust_cooling_schedule()

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if (
            self.metrics.start_irls_iter is None
            and self.invProb.phi_d < self.misfit_from_chi_factor(self.chifact_start)
        ):
            self.start_irls()

        # Perform IRLS (only after `self.cooling_rate` iterations)
        if (
            self.metrics.start_irls_iter is not None
            and (self.opt.iter - self.metrics.start_irls_iter) % self.cooling_rate == 0
        ):
            if self.stopping_criteria():
                self.opt.stopNextIteration = True
                return
            else:
                self.opt.stopNextIteration = False

            # Cool irls thresholds
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                for obj in reg.objfcts:
                    obj.irls_threshold /= self.irls_cooling_factor

            self.metrics.irls_iteration_count += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                reg.update_weights(reg.model)

            self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Apply beta cooling schedule mechanism
        if self.opt.iter > 0 and self.opt.iter % self.cooling_rate == 0:
            self.invProb.beta /= self.cooling_factor

    def start_irls(self):
        """
        Start the IRLS iterations by computing the initial threshold values.
        """
        if self.verbose:
            print(
                "Reached starting chifact with l2-norm regularization:"
                + " Start IRLS steps..."
            )

        self.metrics.start_irls_iter = getattr(self.opt, "iter", 0)
        self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Either use the supplied irls_threshold, or fix base on distribution of
        # model values
        for reg, norms in zip(self.reg.objfcts, self.metrics.input_norms):
            if not isinstance(reg, Sparse):
                continue

            for obj in reg.objfcts:
                threshold = np.percentile(
                    np.abs(obj.mapping * obj._delta_m(self.invProb.model)),
                    self.percentile,
                )
                if isinstance(obj, SmoothnessFirstOrder):
                    threshold /= reg.regularization_mesh.base_length

                obj.irls_threshold = threshold

            reg.norms = norms

            if self.verbose:
                print("irls_threshold " + str(reg.objfcts[0].irls_threshold))

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        self.cooling_factor = 1.0

    def validate(self, directiveList=None):
        directive_list = directiveList.dList
        self_ind = directive_list.index(self)
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in directive_list]

        if any(lin_precond_ind):
            if lin_precond_ind.index(True) < self_ind:
                raise AssertionError(
                    "The directive 'UpdatePreconditioner' must be after UpdateIRLS "
                    "in the directiveList"
                )
        else:
            warnings.warn(
                "Without a Linear preconditioner, convergence may be slow. "
                "Consider adding `directives.UpdatePreconditioner` to your "
                "directives list",
                stacklevel=2,
            )

        beta_schedule = [
            d for d in directive_list if isinstance(d, BetaSchedule) and d is not self
        ]

        if beta_schedule:
            raise AssertionError(
                "Beta scheduling is handled by the `UpdateIRLS` directive."
                "Remove the redundant `BetaSchedule` from your list of directives.",
            )

        spherical_scale = [isinstance(d, SphericalUnitsWeights) for d in directive_list]
        if any(spherical_scale):
            assert spherical_scale.index(True) < self_ind, (
                "The directive 'SphericalUnitsWeights' must be before UpdateIRLS "
                "in the directiveList"
            )

        return True

    def stopping_criteria(self):
        """
        Check for stopping criteria of max_irls_iteration or minimum change.
        """
        phim_new = 0
        for reg in self.reg.objfcts:
            if isinstance(reg, (Sparse, BaseSparse)):
                reg.model = self.invProb.model
                phim_new += reg(reg.model)

        # Check for maximum number of IRLS cycles
        if self.metrics.irls_iteration_count == self.max_irls_iterations:
            if self.verbose:
                print(
                    "Reach maximum number of IRLS cycles:"
                    + f" {self.max_irls_iterations:d}"
                )
            return True

        # Check if the function has changed enough
        f_change = np.abs(self.metrics.f_old - phim_new) / (self.metrics.f_old + 1e-12)

        if (
            f_change < self.f_min_change
            and self.metrics.irls_iteration_count > 1
            and np.abs(
                1.0
                - self.invProb.phi_d / self.misfit_from_chi_factor(self.chifact_target)
            )
            < self.misfit_tolerance
        ):
            if self.verbose:
                print("Minimum decrease in regularization. End of IRLS")
            return True

        self.metrics.f_old = phim_new

        return False


class SphericalUnitsWeights(InversionDirective):
    """
    Directive to update the regularization weights to account for spherical
    parameters in radian and SI.

    The scaling applied to the regularization weights is based on the ratio
    between the maximum value of the model and the maximum value of angles (pi).

    Parameters
    ----------
    amplitude: Projection
        Map to the model parameters for the amplitude of the vector
    angles: list[WeightedLeastSquares]
        List of WeightedLeastSquares for the angles.
    verbose: bool
        Print information to the screen.
    """

    def __init__(
        self,
        amplitude: Projection,
        angles: list[WeightedLeastSquares],
        verbose: bool = True,
        **kwargs,
    ):

        if not isinstance(amplitude, Projection):
            raise TypeError(
                "Attribute 'amplitude' must be of type " "'wires.Projection'"
            )

        self._amplitude = amplitude

        if not isinstance(angles, (list, tuple)) or not all(
            [isinstance(fun, WeightedLeastSquares) for fun in angles]
        ):
            raise TypeError(
                "Attribute 'angles' must be a list of "
                "'regularization.WeightedLeastSquares'."
            )

        self._angles = angles

        super().__init__(
            verbose=verbose,
            **kwargs,
        )

    def initialize(self):
        self.update_scaling()

    def endIter(self):
        self.update_scaling()

    def update_scaling(self):
        """
        Add an 'angle_scale' to the list of weights on the angle regularization for the
        different block of models to account for units of radian and SI.
        """
        amplitude = self._amplitude * self.invProb.model
        max_p = max(amplitude)

        for reg in self._angles:
            for obj in reg.objfcts:
                if obj.units != "radian":
                    continue

                obj.set_weights(angle_scale=np.ones_like(amplitude) * max_p / np.pi)


@deprecate_class(removal_version="0.24.0", error=True)
class Update_IRLS(UpdateIRLS):
    pass
