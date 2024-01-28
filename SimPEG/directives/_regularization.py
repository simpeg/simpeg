import warnings

import numpy as np

from SimPEG.directives import InversionDirective, UpdatePreconditioner, BetaSchedule
from SimPEG.regularization import Sparse, BaseSparse, SmoothnessFirstOrder
from SimPEG.utils import validate_integer, validate_float, validate_type
from SimPEG.utils.code_utils import deprecate_property


class Update_IRLS(InversionDirective):
    """
    Directive to control the IRLS iterations for regularization.Sparse.
    """

    def __init__(
        self,
        max_irls_iterations=20,
        update_beta=True,
        beta_search=False,
        cooling_factor=2.0,
        cooling_rate=1,
        **kwargs,
    ):
        self._input_norms: list[float] | None = None
        self._beta_schedule: BetaSchedule | None = None
        self._target: float | None = None
        self._start: float | None = None

        self.f_old = 0
        self.f_min_change = 1e-2
        self.beta_tol = 1e-1

        self.prctile = 100
        self.chifact_start = 1.0
        self.chifact_target = 1.0

        # Solving parameter for IRLS (mode:2)
        self.irls_iteration = 0
        self.minGNiter = 1
        self.iterStart = 0
        self.sphericalDomain = False

        # Beta schedule
        self.mode = 1
        self.coolEpsFact = 1.2
        self.silent = False

        super().__init__(**kwargs)
        self.max_irls_iterations = max_irls_iterations
        self.update_beta = update_beta
        self.beta_search = beta_search
        self.cooling_factor = cooling_factor
        self.cooling_rate = cooling_rate

    @property
    def max_irls_iterations(self):
        """Maximum irls iterations.

        Returns
        -------
        int
        """
        return self._max_irls_iterations

    @max_irls_iterations.setter
    def max_irls_iterations(self, value):
        self._max_irls_iterations = validate_integer(
            "max_irls_iterations", value, min_val=0
        )

    @property
    def cooling_factor(self):
        """Beta is divided by this value every `cooling_rate` iterations.

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

    coolingFactor = deprecate_property(
        cooling_factor,
        "coolingFactor",
        "cooling_factor",
        future_warn=True,
        removal_version="0.22.0",
    )

    @property
    def cooling_rate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._cooling_rate

    @cooling_rate.setter
    def cooling_rate(self, value):
        self._cooling_rate = validate_integer("cooling_rate", value, min_val=1)

    coolingRate = deprecate_property(
        cooling_rate,
        "coolingRate",
        "cooling_rate",
        future_warn=True,
        removal_version="0.22.0",
    )

    minGNiter = deprecate_property(
        cooling_rate,
        "minGNiter",
        "cooling_rate",
        future_warn=True,
        removal_version="0.22.0",
    )

    @property
    def update_beta(self):
        """Whether to update beta.

        Returns
        -------
        bool
        """
        return self._update_beta

    @update_beta.setter
    def update_beta(self, value):
        self._update_beta = validate_type("update_beta", value, bool)

    @property
    def beta_search(self):
        """Whether to do a beta search.

        Returns
        -------
        bool
        """
        return self._beta_search

    @beta_search.setter
    def beta_search(self, value):
        self._beta_search = validate_type("beta_search", value, bool)

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD

            self._target = nD * 0.5 * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    @property
    def start(self):
        if getattr(self, "_start", None) is None:
            if isinstance(self.survey, list):
                self._start = 0
                for survey in self.survey:
                    self._start += survey.nD * 0.5 * self.chifact_start

            else:
                self._start = self.survey.nD * 0.5 * self.chifact_start
        return self._start

    @start.setter
    def start(self, val):
        self._start = val

    def initialize(self):
        if self.mode == 1:
            self._input_norms = []
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue
                self._input_norms.append(reg.norms)
                reg.norms = [2.0 for _ in reg.objfcts]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue

            reg.model = self.invProb.model

    def endIter(self):
        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if self.mode == 1:
            if self.invProb.phi_d < self.start:
                self.start_irls()
            else:
                return

        # Check if misfit is within the tolerance, otherwise scale beta
        if (
            np.abs(1.0 - self.invProb.phi_d / self.target) > self.beta_tol
            and self.update_beta
        ):
            ratio = self.invProb.phi_d / self.target

            if ratio > 1:
                ratio = np.mean([2.0, ratio])
            else:
                ratio = np.mean([0.75, ratio])

            self.beta_schedule.coolingFactor = ratio

            if self.beta_search:
                print("Beta search step")
                # Re-use previous model and continue with new beta
                self.invProb.model = self.reg.objfcts[0].model
                self.opt.xc = self.reg.objfcts[0].model
                self.opt.iter -= 1
                return

        # Only update after GN iterations
        if (self.opt.iter - self.iterStart) % self.minGNiter == 0:
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
                        obj.irls_threshold = obj.irls_threshold / self.coolEpsFact

            self.irls_iteration += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue

                reg.update_weights(reg.model)

            self.update_beta = True
            self.invProb.phi_m_last = self.reg(self.invProb.model)

    def start_irls(self):
        if not self.silent:
            print(
                "Reached starting chifact with l2-norm regularization:"
                + " Start IRLS steps..."
            )

        self.mode = 2

        if getattr(self.opt, "iter", None) is None:
            self.iterStart = 0
        else:
            self.iterStart = self.opt.iter

        self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Either use the supplied irls_threshold, or fix base on distribution of
        # model values
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue

            for obj in reg.objfcts:
                threshold = np.percentile(
                    np.abs(obj.mapping * obj._delta_m(self.invProb.model)), self.prctile
                )
                if isinstance(obj, SmoothnessFirstOrder):
                    threshold /= reg.regularization_mesh.base_length

                obj.irls_threshold = threshold

        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms in zip(self.reg.objfcts, self._input_norms):
            if not isinstance(reg, Sparse):
                continue
            reg.norms = norms

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        # Print to screen
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue
            if not self.silent:
                print("irls_threshold " + str(reg.objfcts[0].irls_threshold))

    @property
    def beta_schedule(self) -> BetaSchedule:
        """
        The beta schedule used by the directive.
        """
        return self._beta_schedule

    def validate(self, directiveList):
        dList = directiveList.dList
        self_ind = dList.index(self)
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

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

        beta_schedule = [isinstance(d, BetaSchedule) for d in dList]
        if not any(beta_schedule):
            warnings.warn(
                "Handling of the beta scheduling by the `Update_IRLS` directive will be deprecated "
                "in future releases. "
                "A `BetaSchedule` will be added to your directives list, please consider adding it yourself.",
                stacklevel=2,
            )
            self._beta_schedule = BetaSchedule(
                coolingFactor=self.cooling_factor, coolingRate=self.cooling_rate
            )
            directiveList.dList.append(self._beta_schedule)

        if self.sphericalDomain:
            warnings.warn(
                "The scaling of spherical parameters by the `Update_IRLS` directive will be deprecated "
                "in future releases. "
                "A `SphericalDomain` will be added to your directives list, please consider adding it yourself.",
                stacklevel=2,
            )
            directiveList.dList = [SphericalDomain()] + directiveList.dList

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
        if self.irls_iteration == self.max_irls_iterations:
            if not self.silent:
                print(
                    "Reach maximum number of IRLS cycles:"
                    + " {0:d}".format(self.max_irls_iterations)
                )
            return True

        # Check if the function has changed enough
        f_change = np.abs(self.f_old - phim_new) / (self.f_old + 1e-12)

        if (
            f_change < self.f_min_change
            and self.irls_iteration > 1
            and np.abs(1.0 - self.invProb.phi_d / self.target) < self.beta_tol
        ):
            print("Minimum decrease in regularization." + "End of IRLS")
            return True

        self.f_old = phim_new

        return False


class SphericalDomain(InversionDirective):
    """
    Directive to update the regularization weights to account for spherical
    parameters in radian and SI.

    The scaling applied to the regularization weights is based on the ratio
    between the maximum value of the model and the maximum value of angles (pi).
    """

    def initialize(self):
        self.component_scaling()

    def endIter(self):
        self.component_scaling()

    def component_scaling(self):
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
