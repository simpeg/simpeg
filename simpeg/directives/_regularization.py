from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass

from .directives import InversionDirective, UpdatePreconditioner, BetaSchedule
from simpeg.regularization import Sparse, BaseSparse, SmoothnessFirstOrder
from simpeg.utils import validate_integer, validate_float


@dataclass
class IRLSMetrics:
    """
    Data class to store metrics used by the IRLS algorithm.

    Parameters
    ----------
    input_norms : list of floats
        List of norms temporarily stored during the initialization.
    irls_iteration_count : int
        Number of IRLS iterations.
    start_irls_iter : int or None
        Iteration number when the IRLS process started.
    f_old : float
        Previous value of the regularization function.
    """

    input_norms: list[float] = (2.0, 2.0, 2.0, 2.0)
    irls_iteration_count: int = 0
    start_irls_iter: int | None = None
    f_old: float = 0.0


class UpdateIRLS(BetaSchedule):
    """
    Directive to control the IRLS iterations for regularization.Sparse.

    Parameters
    ----------
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
    percentile: int
        Percentile of the function values used to determine the initial IRLS threshold.
    verbose: bool
        Print information to the screen.
    """

    def __init__(
        self,
        coolingRate: int = 1,
        coolingFactor: float = 2.0,
        chifact_start: float = 1.0,
        chifact_target: float = 1.0,
        irls_cooling_factor: float = 1.2,
        f_min_change: float = 1e-2,
        max_irls_iterations: int = 20,
        misfit_tolerance: float = 1e-1,
        percentile: int = 100,
        verbose: bool = True,
        **kwargs,
    ):
        self._metrics: IRLSMetrics | None = None
        self.chifact_start: float = chifact_start
        self.chifact_target: float = chifact_target
        self.irls_cooling_factor: float = irls_cooling_factor
        self.f_min_change: float = f_min_change
        self.max_irls_iterations: int = max_irls_iterations
        self.misfit_tolerance: float = misfit_tolerance
        self.percentile: int = percentile

        super().__init__(
            coolingFactor=coolingFactor,
            coolingRate=coolingRate,
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
        self._misfit_tolerance = validate_integer("misfit_tolerance", value, min_val=0)

    @property
    def percentile(self) -> float:
        """Tolerance on deviation from the target chi factor, as a fractional percent."""
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        self._percentile = validate_integer("percentile", value, min_val=0, max_val=100)

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

        chi_factor: Chi factor to compute the target misfit from.
        """
        value = 0
        if isinstance(self.survey, list):
            for survey in self.survey:
                value += survey.nD * chi_factor

        else:
            value = self.survey.nD * chi_factor

        return value

    def adjust_cooling_schedule(self):
        """
        Adjust the cooling schedule based on the misfit.
        """
        if (
            np.abs(
                1.0
                - self.invProb.phi_d / self.misfit_from_chi_factor(self.chifact_target)
            )
            > self.misfit_tolerance
        ):
            ratio = self.invProb.phi_d / self.misfit_from_chi_factor(
                self.chifact_target
            )

            if ratio > 1:
                ratio = np.mean([2.0, ratio])
            else:
                ratio = np.mean([0.75, ratio])

            self.coolingFactor = ratio

    def initialize(self):
        """
        Initialize the IRLS iterations with l2-norm regularization (mode:1).
        """

        input_norms = []
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue

            input_norms += [reg.norms]
            reg.norms = [2.0 for _ in reg.objfcts]

        self._metrics = IRLSMetrics(
            input_norms=input_norms,
        )

    def endIter(self):
        """
        Check on progress of the inversion and start/update the IRLS process.
        """
        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if (
            self.metrics.start_irls_iter is None
            and self.invProb.phi_d < self.misfit_from_chi_factor(self.chifact_start)
        ):
            self.start_irls()

        # Check if misfit is within the tolerance, otherwise scale beta
        self.adjust_cooling_schedule()

        # Only update after GN iterations
        if (
            self.metrics.start_irls_iter is not None
            and (self.opt.iter - self.metrics.start_irls_iter) % self.coolingRate == 0
        ):
            if self.stopping_criteria():
                self.opt.stopNextIteration = True
                return
            else:
                self.opt.stopNextIteration = False

            # Print to screen
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                for obj in reg.objfcts:
                    if isinstance(reg, (Sparse, BaseSparse)):
                        obj.irls_threshold = (
                            obj.irls_threshold / self.irls_cooling_factor
                        )

            self.metrics.irls_iteration_count += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                reg.update_weights(reg.model)

            self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Repeat beta cooling schedule mechanism
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            self.invProb.beta /= self.coolingFactor

    def start_irls(self):
        if self.verbose:
            print(
                "Reached starting chifact with l2-norm regularization:"
                + " Start IRLS steps..."
            )

        if getattr(self.opt, "iter", None) is None:
            self.metrics.start_irls_iter = 0
        else:
            self.metrics.start_irls_iter = self.opt.iter

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

    def validate(self, directiveList=None):
        directive_list = directiveList.dList
        self_ind = directive_list.index(self)
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in directive_list]

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after Update_IRLS "
                "in the directiveList"
            )
        else:
            warnings.warn(
                "Without a Linear preconditioner, convergence may be slow. "
                "Consider adding `Directives.UpdatePreconditioner` to your "
                "directives list",
                stacklevel=2,
            )

        beta_schedule = [
            d for d in directive_list if isinstance(d, BetaSchedule) and d != self
        ]

        assert len(beta_schedule) == 0, (
            "Beta scheduling is handled by the `UpdateIRLS` directive."
            "Remove the redundant `BetaSchedule` from your list of directives.",
        )

        spherical_scale = [isinstance(d, SphericalDomain) for d in directive_list]
        if any(spherical_scale):
            assert spherical_scale.index(True) < self_ind, (
                "The directive 'SphericalDomain' must be before UpdateIRLS "
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
            print("Minimum decrease in regularization." + "End of IRLS")
            return True

        self.metrics.f_old = phim_new

        return False


class SphericalDomain(InversionDirective):
    """
    Directive to update the regularization weights to account for spherical
    parameters in radian and SI.

    The scaling applied to the regularization weights is based on the ratio
    between the maximum value of the model and the maximum value of angles (pi).
    """

    def initialize(self):
        self.update_scaling()

    def endIter(self):
        self.update_scaling()

    def update_scaling(self):
        """
        Add an 'angle_scale' to the list of weights on the angle regularization for the
        different block of models to account for units of radian and SI.
        """
        # TODO Need to establish a clearer connection between the regularizations
        # and the model blocks. There is an assumption here that the first
        # regularization controls the amplitude.
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            f_m = abs(reg.f_m(self.invProb.model))
            max_p += [np.max(f_m)]

        max_p = max(max_p)

        for reg in self.reg.objfcts:
            for obj in reg.objfcts:
                if obj.units != "radian":
                    continue
                # TODO Need to make weights_shapes a public method
                obj.set_weights(
                    angle_scale=np.ones(obj._weights_shapes[0]) * max_p / np.pi
                )
