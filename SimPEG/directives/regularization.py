import numpy as np
from discretize.utils import mkvc

from .tradeoff_estimator import BaseBetaEstimator
from ..maps import IdentityMap, Wires
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
    BaseSimilarityMeasure,
)
from ..utils import (
    eigenvalue_by_power_iteration,
    deprecate_property,
    validate_string,
    diagEst,
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


class UpdateSensitivityWeights(InversionDirective):
    r"""
    Sensitivity weighting for linear and non-linear least-squares inverse problems.

    This directive computes the root-mean squared sensitivities for the
    forward simulation(s) attached to the inverse problem, then truncates
    and scales the result to create cell weights which are applied in the regularization.
    The underlying theory is provided below in the `Notes` section.

    This directive **requires** that the map for the regularization function is either
    class:`SimPEG.maps.Wires` or class:`SimPEG.maps.Identity`. In other words, the
    sensitivity weighting cannot be applied for parametric inversion. In addition,
    the simulation(s) connected to the inverse problem **must** have a ``getJ`` or
    ``getJtJdiag`` method.

    This directive's place in the :class:`DirectivesList` **must** be
    before any directives which update the preconditioner for the inverse problem
    (i.e. :class:`UpdatePreconditioner`), and **must** be before any directives that
    estimate the starting trade-off parameter (i.e. :class:`EstimateBeta_ByEig`
    and :class:`EstimateBetaMaxDerivative`).

    Parameters
    ----------
    every_iteration : bool
        When ``True``, update sensitivity weighting at every model update; non-linear problems.
        When ``False``, create sensitivity weights for starting model only; linear problems.
    threshold : float
        Threshold value for smallest weighting value.
    threshold_method : {'amplitude', 'global', 'percentile'}
        Threshold method for how `threshold_value` is applied:

            - amplitude:
                the smallest root-mean squared sensitivity is a fractional percent of the largest value; must be between 0 and 1.
            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                the smallest root-mean squared sensitivity is set using percentile threshold; must be between 0 and 100.

    normalization_method : {'maximum', 'min_value', None}
        Normalization method applied to sensitivity weights.

        Options are:

            - maximum:
                sensitivity weights are normalized by the largest value such that the largest weight is equal to 1.
            - minimum:
                sensitivity weights are normalized by the smallest value, after thresholding, such that the smallest weights are equal to 1.
            - ``None``:
                normalization is not applied.

    Notes
    -----
    Let :math:`\mathbf{J}` represent the Jacobian. To create sensitivity weights, root-mean squared (RMS) sensitivities
    :math:`\mathbf{s}` are computed by summing the squares of the rows of the Jacobian:

    .. math::
        \mathbf{s} = \Bigg [ \sum_i \, \mathbf{J_{i, \centerdot }}^2 \, \Bigg ]^{1/2}

    The dynamic range of RMS sensitivities can span many orders of magnitude. When computing sensitivity
    weights, thresholding is generally applied to set a minimum value.

    Thresholding
    ^^^^^^^^^^^^

    If **global** thresholding is applied, we add a constant :math:`\tau` to the RMS sensitivities:

    .. math::
        \mathbf{\tilde{s}} = \mathbf{s} + \tau

    In the case of **percentile** thresholding, we let :math:`s_{\%}` represent a given percentile.
    Thresholding to set a minimum value is applied as follows:

    .. math::
        \tilde{s}_j = \begin{cases}
        s_j \;\; for \;\; s_j \geq s_{\%} \\
        s_{\%} \;\; for \;\; s_j < s_{\%}
        \end{cases}

    If **absolute** thresholding is applied, we define :math:`\eta` as a fractional percent.
    In this case, thresholding is applied as follows:

    .. math::
        \tilde{s}_j = \begin{cases}
        s_j \;\; for \;\; s_j \geq \eta s_{max} \\
        \eta s_{max} \;\; for \;\; s_j < \eta s_{max}
        \end{cases}
    """

    def __init__(
        self,
        every_iteration=False,
        threshold_value=1e-12,
        threshold_method="amplitude",
        normalization_method="maximum",
        **kwargs,
    ):
        if "everyIter" in kwargs.keys():
            warnings.warn(
                "'everyIter' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please use 'every_iteration'.",
                stacklevel=2,
            )
            every_iteration = kwargs.pop("everyIter")

        if "threshold" in kwargs.keys():
            warnings.warn(
                "'threshold' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please use 'threshold_value'.",
                stacklevel=2,
            )
            threshold_value = kwargs.pop("threshold")

        if "normalization" in kwargs.keys():
            warnings.warn(
                "'normalization' property is deprecated and will be removed in SimPEG 0.20.0."
                "Please define normalization using 'normalization_method'.",
                stacklevel=2,
            )
            normalization_method = kwargs.pop("normalization")
            if normalization_method is True:
                normalization_method = "maximum"
            else:
                normalization_method = None

        super().__init__(**kwargs)

        self.every_iteration = every_iteration
        self.threshold_value = threshold_value
        self.threshold_method = threshold_method
        self.normalization_method = normalization_method

    @property
    def every_iteration(self):
        """Update sensitivity weights when model is updated.

        When ``True``, update sensitivity weighting at every model update; non-linear problems.
        When ``False``, create sensitivity weights for starting model only; linear problems.

        Returns
        -------
        bool
        """
        return self._every_iteration

    @every_iteration.setter
    def every_iteration(self, value):
        self._every_iteration = validate_type("every_iteration", value, bool)

    everyIter = deprecate_property(
        every_iteration, "everyIter", "every_iteration", removal_version="0.20.0"
    )

    @property
    def threshold_value(self):
        """Threshold value used to set minimum weighting value.

        The way thresholding is applied to the weighting model depends on the
        `threshold_method` property. The choices for `threshold_method` are:

            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                `threshold_value` is a percentile cutoff; must be between 0 and 100
            - amplitude:
                `threshold_value` is the fractional percent of the largest value; must be between 0 and 1


        Returns
        -------
        float
        """
        return self._threshold_value

    @threshold_value.setter
    def threshold_value(self, value):
        self._threshold_value = validate_float("threshold_value", value, min_val=0.0)

    threshold = deprecate_property(
        threshold_value, "threshold", "threshold_value", removal_version="0.20.0"
    )

    @property
    def threshold_method(self):
        """Threshold method for how `threshold_value` is applied:

            - global:
                `threshold_value` is added to the cell weights prior to normalization; must be greater than 0.
            - percentile:
                the smallest root-mean squared sensitivity is set using percentile threshold; must be between 0 and 100
            - amplitude:
                the smallest root-mean squared sensitivity is a fractional percent of the largest value; must be between 0 and 1


        Returns
        -------
        str
        """
        return self._threshold_method

    @threshold_method.setter
    def threshold_method(self, value):
        self._threshold_method = validate_string(
            "threshold_method", value, string_list=["global", "percentile", "amplitude"]
        )

    @property
    def normalization_method(self):
        """Normalization method applied to sensitivity weights.

        Options are:

            - ``None``
                normalization is not applied
            - maximum:
                sensitivity weights are normalized by the largest value such that the largest weight is equal to 1.
            - minimum:
                sensitivity weights are normalized by the smallest value, after thresholding, such that the smallest weights are equal to 1.

        Returns
        -------
        None, str
        """
        return self._normalization_method

    @normalization_method.setter
    def normalization_method(self, value):
        if value is None:
            self._normalization_method = value

        elif isinstance(value, bool):
            warnings.warn(
                "Boolean type for 'normalization_method' is deprecated and will be removed in 0.20.0."
                "Please use None, 'maximum' or 'minimum'.",
                stacklevel=2,
            )
            if value:
                self._normalization_method = "maximum"
            else:
                self._normalization_method = None

        else:
            self._normalization_method = validate_string(
                "normalization_method", value, string_list=["minimum", "maximum"]
            )

    normalization = deprecate_property(
        normalization_method,
        "normalization",
        "normalization_method",
        removal_version="0.20.0",
    )

    def initialize(self):
        """Compute sensitivity weights upon starting the inversion."""
        for reg in self.reg.objfcts:
            if not isinstance(reg.mapping, (IdentityMap, Wires)):
                raise TypeError(
                    f"Mapping for the regularization must be of type {IdentityMap} or {Wires}. "
                    + f"Input mapping of type {type(reg.mapping)}."
                )

        self.update()

    def endIter(self):
        """Execute end of iteration."""

        if self.every_iteration:
            self.update()

    def update(self):
        """Update sensitivity weights"""

        jtj_diag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):
            if getattr(sim, "getJtJdiag", None) is None:
                if getattr(sim, "getJ", None) is None:
                    raise AttributeError(
                        "Simulation does not have a getJ attribute."
                        + "Cannot form the sensitivity explicitly"
                    )
                jtj_diag += mkvc(np.sum((dmisfit.W * sim.getJ(m)) ** 2.0, axis=0))
            else:
                jtj_diag += sim.getJtJdiag(m, W=dmisfit.W)

        # Compute and sum root-mean squared sensitivities for all objective functions
        wr = np.zeros_like(self.invProb.model)
        for reg in self.reg.objfcts:
            if isinstance(reg, BaseSimilarityMeasure):
                continue

            mesh = reg.regularization_mesh
            n_cells = mesh.nC
            mapped_jtj_diag = reg.mapping * jtj_diag
            # reshape the mapped, so you can divide by volume
            # (let's say it was a vector or anisotropic model)
            mapped_jtj_diag = mapped_jtj_diag.reshape((n_cells, -1), order="F")
            wr_temp = mapped_jtj_diag / reg.regularization_mesh.vol[:, None] ** 2.0
            wr_temp = wr_temp.reshape(-1, order="F")

            wr += reg.mapping.deriv(self.invProb.model).T * wr_temp

        wr **= 0.5

        # Apply thresholding
        if self.threshold_method == "global":
            wr += self.threshold_value
        elif self.threshold_method == "percentile":
            wr = np.clip(
                wr, a_min=np.percentile(wr, self.threshold_value), a_max=np.inf
            )
        else:
            wr = np.clip(wr, a_min=self.threshold_value * wr.max(), a_max=np.inf)

        # Apply normalization
        if self.normalization_method == "maximum":
            wr /= wr.max()
        elif self.normalization_method == "minimum":
            wr /= wr.min()

        # Add sensitivity weighting to all model objective functions
        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                sub_regs = getattr(reg, "objfcts", [reg])
                for sub_reg in sub_regs:
                    sub_reg.set_weights(sensitivity=sub_reg.mapping * wr)

    def validate(self, directiveList):
        """Validate directive against directives list.

        The ``UpdateSensitivityWeights`` directive impacts the regularization by applying
        cell weights. As a result, its place in the :class:`DirectivesList` must be
        before any directives which update the preconditioner for the inverse problem
        (i.e. :class:`UpdatePreconditioner`), and must be before any directives that
        estimate the starting trade-off parameter (i.e. :class:`EstimateBeta_ByEig`
        and :class:`EstimateBetaMaxDerivative`).


        Returns
        -------
        bool
            Returns ``True`` if validation passes. Otherwise, an error is thrown.
        """
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)

        beta_estimator_ind = [isinstance(d, BaseBetaEstimator) for d in dList]
        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

        if any(beta_estimator_ind):
            assert beta_estimator_ind.index(True) > self_ind, (
                "The directive for setting intial beta must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        return True


class Update_Wj(InversionDirective):
    """
    Create approx-sensitivity base weighting using the probing method
    """

    def __init__(self, k=None, itr=None, **kwargs):
        self.k = k
        self.itr = itr
        super().__init__(**kwargs)

    @property
    def k(self):
        """Number of probing cycles for the estimator.

        Returns
        -------
        int
        """
        return self._k

    @k.setter
    def k(self, value):
        if value is not None:
            value = validate_integer("k", value, min_val=1)
        self._k = value

    @property
    def itr(self):
        """Which iteration to update the sensitivity.

        Will always update if `None`.

        Returns
        -------
        int or None
        """
        return self._itr

    @itr.setter
    def itr(self, value):
        if value is not None:
            value = validate_integer("itr", value, min_val=1)
        self._itr = value

    def endIter(self):
        if self.itr is None or self.itr == self.opt.iter:
            m = self.invProb.model
            if self.k is None:
                self.k = int(self.survey.nD / 10)

            def JtJv(v):
                Jv = self.simulation.Jvec(m, v)

                return self.simulation.Jtvec(m, Jv)

            JtJdiag = diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag
