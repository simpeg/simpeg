import numpy as np
from ..objective_function import ComboObjectiveFunction
import warnings

from ..regularization import (
    BaseSparse,
    Smallness,
    Sparse,
    SparseSmallness,
    PGIsmallness,
    PGIwithNonlinearRelationshipsSmallness,
    SmoothnessFirstOrder,
    SparseSmoothness,
)
from ..utils import (
    eigenvalue_by_power_iteration,
)
from ..utils.code_utils import (
    validate_integer,
    validate_float,
    validate_ndarray_with_shape,
    validate_type,
)
from .base import InversionDirective
from .optimization import UpdatePreconditioner


class AlphasSmoothEstimate_ByEig(InversionDirective):
    """
    Estimate the alphas multipliers for the smoothness terms of the regularization
    as a multiple of the ratio between the highest eigenvalue of the
    smallness term and the highest eigenvalue of each smoothness term of the regularization.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    def __init__(self, alpha0_ratio=1.0, n_pw_iter=4, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha0_ratio = alpha0_ratio
        self.n_pw_iter = n_pw_iter
        self.seed = seed

    @property
    def alpha0_ratio(self):
        """the estimated Alpha_smooth is multiplied by this ratio (int or array).

        Returns
        -------
        numpy.ndarray
        """
        return self._alpha0_ratio

    @alpha0_ratio.setter
    def alpha0_ratio(self, value):
        self._alpha0_ratio = validate_ndarray_with_shape(
            "alpha0_ratio", value, shape=("*",)
        )

    @property
    def n_pw_iter(self):
        """Number of power iterations for estimation.

        Returns
        -------
        int
        """
        return self._n_pw_iter

    @n_pw_iter.setter
    def n_pw_iter(self, value):
        self._n_pw_iter = validate_integer("n_pw_iter", value, min_val=1)

    @property
    def seed(self):
        """Random seed to initialize with.

        Returns
        -------
        int
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        if value is not None:
            value = validate_integer("seed", value, min_val=1)
        self._seed = value

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        smoothness = []
        smallness = []
        parents = {}
        for regobjcts in self.reg.objfcts:
            if isinstance(regobjcts, ComboObjectiveFunction):
                objfcts = regobjcts.objfcts
            else:
                objfcts = [regobjcts]

            for obj in objfcts:
                if isinstance(
                    obj,
                    (
                        Smallness,
                        SparseSmallness,
                        PGIsmallness,
                        PGIwithNonlinearRelationshipsSmallness,
                    ),
                ):
                    smallness += [obj]

                elif isinstance(obj, (SmoothnessFirstOrder, SparseSmoothness)):
                    parents[obj] = regobjcts
                    smoothness += [obj]

        if len(smallness) == 0:
            raise UserWarning(
                "Directive 'AlphasSmoothEstimate_ByEig' requires a regularization with at least one Small instance."
            )

        smallness_eigenvalue = eigenvalue_by_power_iteration(
            smallness[0],
            self.invProb.model,
            n_pw_iter=self.n_pw_iter,
        )

        self.alpha0_ratio = self.alpha0_ratio * np.ones(len(smoothness))

        if len(self.alpha0_ratio) != len(smoothness):
            raise ValueError(
                f"Input values for 'alpha0_ratio' should be of len({len(smoothness)}). Provided {self.alpha0_ratio}"
            )

        alphas = []
        for user_alpha, obj in zip(self.alpha0_ratio, smoothness):
            smooth_i_eigenvalue = eigenvalue_by_power_iteration(
                obj,
                self.invProb.model,
                n_pw_iter=self.n_pw_iter,
            )
            ratio = smallness_eigenvalue / smooth_i_eigenvalue

            mtype = obj._multiplier_pair

            new_alpha = getattr(parents[obj], mtype) * user_alpha * ratio
            setattr(parents[obj], mtype, new_alpha)
            alphas += [new_alpha]

        if self.verbose:
            print(f"Alpha scales: {alphas}")


class Update_IRLS(InversionDirective):
    f_old = 0
    f_min_change = 1e-2
    beta_tol = 1e-1
    beta_ratio_l2 = None
    prctile = 100
    chifact_start = 1.0
    chifact_target = 1.0

    # Solving parameter for IRLS (mode:2)
    irls_iteration = 0
    minGNiter = 1
    iterStart = 0
    sphericalDomain = False

    # Beta schedule
    ComboObjFun = False
    mode = 1
    coolEpsOptimized = True
    coolEps_p = True
    coolEps_q = True
    floorEps_p = 1e-8
    floorEps_q = 1e-8
    coolEpsFact = 1.2
    silent = False
    fix_Jmatrix = False

    def __init__(
        self,
        max_irls_iterations=20,
        update_beta=True,
        beta_search=False,
        coolingFactor=2.0,
        coolingRate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_irls_iterations = max_irls_iterations
        self.update_beta = update_beta
        self.beta_search = beta_search
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

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
    def coolingFactor(self):
        """Beta is divided by this value every `coolingRate` iterations.

        Returns
        -------
        float
        """
        return self._coolingFactor

    @coolingFactor.setter
    def coolingFactor(self, value):
        self._coolingFactor = validate_float(
            "coolingFactor", value, min_val=0.0, inclusive_min=False
        )

    @property
    def coolingRate(self):
        """Cool after this number of iterations.

        Returns
        -------
        int
        """
        return self._coolingRate

    @coolingRate.setter
    def coolingRate(self, value):
        self._coolingRate = validate_integer("coolingRate", value, min_val=1)

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
            self.norms = []
            for reg in self.reg.objfcts:
                if not isinstance(reg, Sparse):
                    continue
                self.norms.append(reg.norms)
                reg.norms = [2.0 for obj in reg.objfcts]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
            if not isinstance(reg, Sparse):
                continue

            reg.model = self.invProb.model

        if self.sphericalDomain:
            self.angleScale()

    def endIter(self):
        if self.sphericalDomain:
            self.angleScale()

        # Check if misfit is within the tolerance, otherwise scale beta
        if np.all(
            [
                np.abs(1.0 - self.invProb.phi_d / self.target) > self.beta_tol,
                self.update_beta,
                self.mode != 1,
            ]
        ):
            ratio = self.target / self.invProb.phi_d

            if ratio > 1:
                ratio = np.mean([2.0, ratio])
            else:
                ratio = np.mean([0.75, ratio])

            self.invProb.beta = self.invProb.beta * ratio

            if np.all([self.mode != 1, self.beta_search]):
                print("Beta search step")
                # self.update_beta = False
                # Re-use previous model and continue with new beta
                self.invProb.model = self.reg.objfcts[0].model
                self.opt.xc = self.reg.objfcts[0].model
                self.opt.iter -= 1
                return

        elif np.all([self.mode == 1, self.opt.iter % self.coolingRate == 0]):
            self.invProb.beta = self.invProb.beta / self.coolingFactor

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.start, self.mode == 1]):
            self.start_irls()

        # Only update after GN iterations
        if np.all(
            [(self.opt.iter - self.iterStart) % self.minGNiter == 0, self.mode != 1]
        ):
            if self.stopping_criteria():
                self.opt.stopNextIteration = True
                return

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
        for reg, norms in zip(self.reg.objfcts, self.norms):
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

    def angleScale(self):
        """
        Update the scales used by regularization for the
        different block of models
        """
        # Currently implemented for MVI-S only
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            f_m = abs(reg.f_m(reg.model))
            max_p += [np.max(f_m)]

        max_p = np.asarray(max_p).max()

        max_s = [np.pi, np.pi]

        for reg, var in zip(self.reg.objfcts[1:], max_s):
            for obj in reg.objfcts:
                # TODO Need to make weights_shapes a public method
                obj.set_weights(
                    angle_scale=np.ones(obj._weights_shapes[0]) * max_p / var
                )

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
        if np.all(
            [
                f_change < self.f_min_change,
                self.irls_iteration > 1,
                np.abs(1.0 - self.invProb.phi_d / self.target) < self.beta_tol,
            ]
        ):
            print("Minimum decrease in regularization." + "End of IRLS")
            return True

        self.f_old = phim_new

        return False
