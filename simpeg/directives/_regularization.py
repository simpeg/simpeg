from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass
from typing import Literal
from .directives import InversionDirective, UpdatePreconditioner, BetaSchedule
from simpeg.regularization import Sparse, BaseSparse, SmoothnessFirstOrder
from simpeg.utils import validate_integer, validate_float
from simpeg.utils.code_utils import deprecate_property


@dataclass
class IRLSMetrics:
    """
    Data class to store metrics used by the IRLS algorithm.

    Parameters
    ----------

    input_norms: List of norms temporarily stored during the initialization.
    irls_iteration_count: Number of IRLS iterations.
    start_irls_iter: Iteration number when the IRLS process started.
    f_old: Previous value of the regularization function.
    mode: Mode of the IRLS algorithm, 1 for l2-norm and 2 for lp-norm iterations.
    """

    input_norms: list[float]
    irls_iteration_count = 0
    start_irls_iter = 0
    f_old: float = 0.0
    mode: Literal[1, 2] = 1


class UpdateIRLS(InversionDirective):
    """
    Directive to control the IRLS iterations for regularization.Sparse.

    Parameters
    ----------

    beta_search: bool
        Proceed with a beta search step if the target misfit is outside the tolerance.
    misfit_tolerance: float
        Tolerance for the target misfit.
    chifact_start: float
        Starting chi factor for the IRLS iterations.
    chifact_target: float
        Target chi factor for the IRLS iterations.
    cooling_factor: float
        Factor to cool the IRLS threshold epsilon.
    f_min_change: float
        Minimum change in the regularization function to continue the IRLS iterations.
    max_irls_iterations: int
        Maximum number of IRLS iterations.
    percentile: float
        Percentile of the function values used to determine the initial IRLS threshold.
    """

    def __init__(
        self,
        beta_schedule: BetaSchedule | None = None,
        chifact_start: float = 1.0,
        chifact_target: float = 1.0,
        cooling_factor: float = 1.2,
        f_min_change: float = 1e-2,
        max_irls_iterations: int = 20,
        misfit_tolerance: float = 1e-1,
        percentile: int = 100,
        **kwargs,
    ):
        self._metrics: IRLSMetrics | None = None
        self.beta_schedule: BetaSchedule = beta_schedule
        self.chifact_start: float = chifact_start
        self.chifact_target: float = chifact_target
        self.cooling_factor: float = cooling_factor
        self.f_min_change: float = f_min_change
        self.max_irls_iterations: int = max_irls_iterations
        self.misfit_tolerance: float = misfit_tolerance
        self.percentile: int = percentile

        if "silent" in kwargs:
            warnings.warn(
                "The `silent` keyword argument is deprecated. "
                "Use `verbose` instead.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs["verbose"] = not kwargs.pop("silent")

        super().__init__(**kwargs)

    @property
    def metrics(self) -> IRLSMetrics:
        """Various metrics used by the IRLS algorithm."""
        if self._metrics is None:
            self._metrics = IRLSMetrics(
                input_norms=self.get_input_norms(),
            )
        return self._metrics

    def get_input_norms(self) -> list[float]:
        input_norms = []
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue
            input_norms += [reg.norms]

        return input_norms

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

    beta_tol = deprecate_property(
        misfit_tolerance,
        "beta_tol",
        "UpdateIRLS.misfit_tolerance",
        future_warn=True,
        removal_version="0.22.0",
    )

    @property
    def percentile(self) -> float:
        """Tolerance on deviation from the target chi factor, as a fractional percent."""
        return self._percentile

    @percentile.setter
    def percentile(self, value):
        self._percentile = validate_integer("percentile", value, min_val=0, max_val=100)

    prctile = deprecate_property(
        percentile,
        "prctile",
        "UpdateIRLS.percentile",
        future_warn=True,
        removal_version="0.22.0",
    )

    @property
    def iterStart(self) -> int:
        """Iteration number when the IRLS process started."""

        warnings.warn(
            "The `iterStart` property is deprecated. Use `metrics.start_irls_iter` instead.",
            stacklevel=2,
        )

        return self.metrics.start_irls_iter

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
    def cooling_factor(self) -> float:
        """IRLS threshold parameter (epsilon) is divided by this value every iteration."""
        return self._cooling_factor

    @cooling_factor.setter
    def cooling_factor(self, value):
        self._cooling_factor = validate_float(
            "cooling_factor", value, min_val=0.0, inclusive_min=False
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

    def initialize(self):
        """
        Initialize the IRLS iterations with l2-norm regularization (mode:1).
        """
        if self.metrics.mode == 1:
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue
                reg.norms = [2.0 for _ in reg.objfcts]

    def endIter(self):
        """
        Check on progress of the inversion and start/update the IRLS process.
        """
        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if self.metrics.mode == 1:
            if self.invProb.phi_d < self.misfit_from_chi_factor(self.chifact_start):
                self.start_irls()
            else:
                return

        # Check if misfit is within the tolerance, otherwise scale beta
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

            self.beta_schedule.coolingFactor = ratio

        # Only update after GN iterations
        if (
            self.opt.iter - self.metrics.start_irls_iter
        ) % self.beta_schedule.coolingRate == 0:
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
                        obj.irls_threshold = obj.irls_threshold / self.cooling_factor

            self.metrics.irls_iteration_count += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                reg.update_weights(reg.model)

            self.invProb.phi_m_last = self.reg(self.invProb.model)

        self.beta_schedule.endIter()

    def start_irls(self):
        if self.verbose:
            print(
                "Reached starting chifact with l2-norm regularization:"
                + " Start IRLS steps..."
            )

        self.metrics.mode = 2

        if getattr(self.opt, "iter", None) is None:
            self.metrics.start_irls_iter = 0
        else:
            self.metrics.start_irls_iter = self.opt.iter

        self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Either use the supplied irls_threshold, or fix base on distribution of
        # model values
        for reg in self.reg.objfcts:
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

        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms in zip(self.reg.objfcts, self.metrics.input_norms):
            if not isinstance(reg, Sparse):
                continue
            reg.norms = norms

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        # Print to screen
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue
            if self.verbose:
                print("irls_threshold " + str(reg.objfcts[0].irls_threshold))

    @property
    def beta_schedule(self) -> BetaSchedule:
        """
        The beta schedule used by the directive.
        """
        return self._beta_schedule

    @beta_schedule.setter
    def beta_schedule(self, directive):
        """
        The beta schedule used by the directive.
        """
        if directive is None:
            directive = BetaSchedule(coolingFactor=2.0, coolingRate=1)

        if not isinstance(directive, BetaSchedule):
            raise TypeError("The beta schedule must be an instance of `BetaSchedule`.")

        self._beta_schedule = directive

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

        beta_schedule = [d for d in directive_list if isinstance(d, BetaSchedule)]

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

        # Check for maximum number of IRLS cycles1
        if self.metrics.irls_iteration_count == self.max_irls_iterations:
            if self.verbose:
                print(
                    "Reach maximum number of IRLS cycles:"
                    + " {0:d}".format(self.max_irls_iterations)
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

        max_p = np.asarray(max_p).max()

        for reg in self.reg.objfcts:
            for obj in reg.objfcts:
                if obj.units != "radian":
                    continue
                # TODO Need to make weights_shapes a public method
                obj.set_weights(
                    angle_scale=np.ones(obj._weights_shapes[0]) * max_p / np.pi
                )
