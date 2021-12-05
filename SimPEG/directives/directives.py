import properties
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import scipy.sparse as sp
from ..data_misfit import BaseDataMisfit
from ..objective_function import ComboObjectiveFunction
from ..maps import IdentityMap, Wires
from ..regularization import (
    BaseComboRegularization,
    BaseRegularization,
    SimpleSmall,
    Small,
    SparseSmall,
    Simple,
    Tikhonov,
    Sparse,
    PGIsmallness,
    PGIwithNonlinearRelationshipsSmallness,
    PGI,
    SmoothDeriv,
    SimpleSmoothDeriv,
    SparseDeriv,
    PGIwithRelationships,
    BaseSimilarityMeasure,
)
from ..utils import (
    mkvc,
    setKwargs,
    sdiag,
    diagEst,
    spherical2cartesian,
    cartesian2spherical,
    Zero,
    eigenvalue_by_power_iteration,
)
from ..utils.code_utils import deprecate_property
from .. import optimization


class InversionDirective(properties.HasProperties):
    """InversionDirective"""

    _REGISTRY = {}

    debug = False  #: Print debugging information
    _regPair = [BaseComboRegularization, BaseRegularization, ComboObjectiveFunction]
    _dmisfitPair = [BaseDataMisfit, ComboObjectiveFunction]

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self, "_inversion", None)

    @inversion.setter
    def inversion(self, i):
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "InversionDirective {0!s} has switched to a new inversion.".format(
                    self.__class__.__name__
                )
            )
        self._inversion = i

    @property
    def invProb(self):
        return self.inversion.invProb

    @property
    def opt(self):
        return self.invProb.opt

    @property
    def reg(self):
        if getattr(self, "_reg", None) is None:
            self.reg = self.invProb.reg  # go through the setter
        return self._reg

    @reg.setter
    def reg(self, value):
        assert any(
            [isinstance(value, regtype) for regtype in self._regPair]
        ), "Regularization must be in {}, not {}".format(self._regPair, type(value))

        if isinstance(value, BaseComboRegularization):
            value = 1 * value  # turn it into a combo objective function
        self._reg = value

    @property
    def dmisfit(self):
        if getattr(self, "_dmisfit", None) is None:
            self.dmisfit = self.invProb.dmisfit  # go through the setter
        return self._dmisfit

    @dmisfit.setter
    def dmisfit(self, value):

        assert any(
            [isinstance(value, dmisfittype) for dmisfittype in self._dmisfitPair]
        ), "Misfit must be in {}, not {}".format(self._dmisfitPair, type(value))

        if not isinstance(value, ComboObjectiveFunction):
            value = 1 * value  # turn it into a combo objective function
        self._dmisfit = value

    @property
    def survey(self):
        """
        Assuming that dmisfit is always a ComboObjectiveFunction,
        return a list of surveys for each dmisfit [survey1, survey2, ... ]
        """
        return [objfcts.simulation.survey for objfcts in self.dmisfit.objfcts]

    @property
    def simulation(self):
        """
        Assuming that dmisfit is always a ComboObjectiveFunction,
        return a list of problems for each dmisfit [prob1, prob2, ...]
        """
        return [objfcts.simulation for objfcts in self.dmisfit.objfcts]

    prob = deprecate_property(
        simulation,
        "prob",
        new_name="simulation",
        removal_version="0.16.0",
        future_warn=True,
    )

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass

    def validate(self, directiveList=None):
        return True


class DirectiveList(object):

    dList = None  #: The list of Directives

    def __init__(self, *directives, **kwargs):
        self.dList = []
        for d in directives:
            assert isinstance(
                d, InversionDirective
            ), "All directives must be InversionDirectives not {}".format(type(d))
            self.dList.append(d)
        setKwargs(self, **kwargs)

    @property
    def debug(self):
        return getattr(self, "_debug", False)

    @debug.setter
    def debug(self, value):
        for d in self.dList:
            d.debug = value
        self._debug = value

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self, "_inversion", None)

    @inversion.setter
    def inversion(self, i):
        if self.inversion is i:
            return
        if getattr(self, "_inversion", None) is not None:
            warnings.warn(
                "{0!s} has switched to a new inversion.".format(self.__class__.__name__)
            )
        for d in self.dList:
            d.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.dList is None:
            if self.debug:
                print("DirectiveList is None, no directives to call!")
            return

        directives = ["initialize", "endIter", "finish"]
        assert ruleType in directives, 'Directive type must be in ["{0!s}"]'.format(
            '", "'.join(directives)
        )
        for r in self.dList:
            getattr(r, ruleType)()

    def validate(self):
        [directive.validate(self) for directive in self.dList]
        return True


class BetaEstimate_ByEig(InversionDirective):
    """
    Estimate the trade-off parameter beta between the data misfit(s) and the
    regularization as a multiple of the ratio between the highest eigenvalue of the
    data misfit term and the highest eigenvalue of the regularization.
    The highest eigenvalues are estimated through power iterations and Rayleigh quotient.

    """

    beta0_ratio = 1.0  #: the estimated ratio is multiplied by this to obtain beta
    n_pw_iter = 4  #: number of power iterations for estimation.
    seed = None  #: Random seed for the directive

    def initialize(self):
        """
        The initial beta is calculated by comparing the estimated
        eigenvalues of JtJ and WtW.
        To estimate the eigenvector of **A**, we will use one iteration
        of the *Power Method*:

        .. math::
            \mathbf{x_1 = A x_0}

        Given this (very course) approximation of the eigenvector, we can
        use the *Rayleigh quotient* to approximate the largest eigenvalue.

        .. math::
            \lambda_0 = \\frac{\mathbf{x^\\top A x}}{\mathbf{x^\\top x}}

        We will approximate the largest eigenvalue for both JtJ and WtW,
        and use some ratio of the quotient to estimate beta0.

        .. math::
            \\beta_0 = \gamma \\frac{\mathbf{x^\\top J^\\top J x}}{\mathbf{x^\\top W^\\top W x}}

        :rtype: float
        :return: beta0
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model

        dm_eigenvalue = eigenvalue_by_power_iteration(
            self.dmisfit, m, n_pw_iter=self.n_pw_iter,
        )
        reg_eigenvalue = eigenvalue_by_power_iteration(
            self.reg, m, n_pw_iter=self.n_pw_iter,
        )

        self.ratio = dm_eigenvalue / reg_eigenvalue
        self.beta0 = self.beta0_ratio * self.ratio

        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """
        Directive for beta cooling schedule to determine the tradeoff
        parameters of the joint inverse problem.
        We borrow some code from Update_IRLS.
    """

    chifact_target = 1.0
    beta_tol = 1e-1
    update_beta = True
    coolingRate = 3
    coolingFactor = 8
    dmis_met = False

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

    def endIter(self):

        if self.invProb.phi_d < self.target:
            self.dmis_met = True

        if np.all(
            [
                np.abs(1.0 - self.invProb.phi_d / self.target) > self.beta_tol,
                self.update_beta,
                self.dmis_met,
                self.opt.iter % self.coolingRate == 0,
            ]
        ):

            ratio = self.target / self.invProb.phi_d

            if ratio > 1:
                ratio = np.minimum(1.5, ratio)
            else:
                ratio = np.maximum(0.75, ratio)

            self.invProb.beta = self.invProb.beta * ratio

        elif np.all([self.opt.iter % self.coolingRate == 0, self.dmis_met == False]):
            self.invProb.beta = self.invProb.beta / self.coolingFactor


class AlphasSmoothEstimate_ByEig(InversionDirective):
    """
    Estimate the alphas multipliers for the smoothness terms of the regularization
    as a multiple of the ratio between the highest eigenvalue of the
    smallness term and the highest eigenvalue of each smoothness term of the regularization.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    alpha0_ratio = (
        1.0  #: the estimated Alpha_smooth is multiplied by this ratio (int or array)
    )
    n_pw_iter = 4  #: number of power iterations for the estimate
    verbose = False  #: print the estimated alphas at the initialization
    debug = False  #: print the current process
    seed = None  # random seed for the directive

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        if getattr(self.reg.objfcts[0], "objfcts", None) is not None:
            nbr = np.sum(
                [len(self.reg.objfcts[i].objfcts) for i in range(len(self.reg.objfcts))]
            )
            # Find the smallness terms in a two-levels combo-regularization.
            smallness = []
            alpha0 = []
            for i, regobjcts in enumerate(self.reg.objfcts):
                for j, regpart in enumerate(regobjcts.objfcts):
                    alpha0 += [self.reg.multipliers[i] * regobjcts.multipliers[j]]
                    smallness += [
                        [
                            i,
                            j,
                            isinstance(
                                regpart,
                                (
                                    SimpleSmall,
                                    Small,
                                    SparseSmall,
                                    PGIsmallness,
                                    PGIwithNonlinearRelationshipsSmallness,
                                ),
                            ),
                        ]
                    ]
            smallness = np.r_[smallness]
            # Select the first, only considered, smallness term.
            smallness = smallness[smallness[:, 2] == 1][:, :2][0]

            # Find the smoothness terms in a two-levels combo-regularization.
            smoothness = []
            for i, regobjcts in enumerate(self.reg.objfcts):
                for j, regpart in enumerate(regobjcts.objfcts):
                    smoothness += [
                        [
                            i,
                            j,
                            isinstance(
                                regpart, (SmoothDeriv, SimpleSmoothDeriv, SparseDeriv)
                            ),
                        ]
                    ]
            smoothness = np.r_[smoothness]
            mode = 1

        else:
            nbr = len(self.reg.objfcts)
            alpha0 = self.reg.multipliers
            smoothness = np.r_[
                [
                    isinstance(regpart, (SmoothDeriv, SimpleSmoothDeriv, SparseDeriv))
                    for regpart in self.reg.objfcts
                ]
            ]
            mode = 2

        if not isinstance(self.alpha0_ratio, np.ndarray):
            self.alpha0_ratio = self.alpha0_ratio * np.ones(nbr)

        if self.debug:
            print("Calculating the Alpha0 parameter.")

        m = self.invProb.model

        if mode == 2:
            smallness_eigenvalue = eigenvalue_by_power_iteration(
                self.reg.objfcts[0], m, n_pw_iter=self.n_pw_iter,
            )
            for i in range(nbr):
                if smoothness[i]:
                    smooth_i_eigenvalue = eigenvalue_by_power_iteration(
                        self.reg.objfcts[i], m, n_pw_iter=self.n_pw_iter,
                    )
                    ratio = smallness_eigenvalue / smooth_i_eigenvalue

                    alpha0[i] *= self.alpha0_ratio[i] * ratio
                    mtype = self.reg.objfcts[i]._multiplier_pair
                    setattr(self.reg, mtype, alpha0[i])

        elif mode == 1:
            smallness_eigenvalue = eigenvalue_by_power_iteration(
                self.reg.objfcts[smallness[0]].objfcts[smallness[1]],
                m,
                n_pw_iter=self.n_pw_iter,
            )
            for i in range(nbr):
                ratio = []
                if smoothness[i, 2]:
                    idx = smoothness[i, :2]
                    smooth_i_eigenvalue = eigenvalue_by_power_iteration(
                        self.reg.objfcts[idx[0]].objfcts[idx[1]],
                        m,
                        n_pw_iter=self.n_pw_iter,
                    )

                    ratio = np.divide(
                        smallness_eigenvalue,
                        smooth_i_eigenvalue,
                        out=np.zeros_like(smallness_eigenvalue),
                        where=smooth_i_eigenvalue != 0,
                    )

                    alpha0[i] *= self.alpha0_ratio[i] * ratio
                    mtype = self.reg.objfcts[idx[0]].objfcts[idx[1]]._multiplier_pair
                    setattr(self.reg.objfcts[idx[0]], mtype, alpha0[i])

        if self.verbose:
            print("Alpha scales: ", self.reg.multipliers)
            if mode == 1:
                for objf in self.reg.objfcts:
                    print("Alpha scales: ", objf.multipliers)


class ScalingMultipleDataMisfits_ByEig(InversionDirective):
    """
    For multiple data misfits only: multiply each data misfit term
    by the inverse of its highest eigenvalue and then
    normalize the sum of the data misfit multipliers to one.
    The highest eigenvalue are estimated through power iterations and Rayleigh quotient.
    """

    n_pw_iter = 4  #: number of power iterations for the estimate
    chi0_ratio = None  #: The initial scaling ratio (default is data misfit multipliers)
    verbose = False  #: print the estimated data misfits multipliers
    debug = False  #: print the current process
    seed = None  # random seed for the directive

    def initialize(self):
        """"""
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("Calculating the scaling parameter.")

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError(
                "ScalingMultipleDataMisfits_ByEig only applies to joint inversion"
            )

        ndm = len(self.dmisfit.objfcts)
        if self.chi0_ratio is not None:
            self.chi0_ratio = self.chi0_ratio * np.ones(ndm)
        else:
            self.chi0_ratio = self.dmisfit.multipliers

        m = self.invProb.model

        dm_eigenvalue_list = []
        for j, dm in enumerate(self.dmisfit.objfcts):
            dm_eigenvalue_list += [eigenvalue_by_power_iteration(dm, m)]

        self.chi0 = self.chi0_ratio / np.r_[dm_eigenvalue_list]
        self.chi0 = self.chi0 / np.sum(self.chi0)
        self.dmisfit.multipliers = self.chi0

        if self.verbose:
            print("Scale Multipliers: ", self.dmisfit.multipliers)


class JointScalingSchedule(InversionDirective):
    """
    For multiple data misfits only: rebalance each data misfit term
    during the inversion when some datasets are fit, and others not
    using the ratios of current misfits and their respective target.
    It implements the strategy described in https://doi.org/10.1093/gji/ggaa378.
    """

    verbose = False
    warmingFactor = 1.0
    mode = 1
    chimax = 1e10
    chimin = 1e-10
    update_rate = 1

    def initialize(self):

        if (
            getattr(self.dmisfit, "objfcts", None) is None
            or len(self.dmisfit.objfcts) == 1
        ):
            raise TypeError("JointScalingSchedule only applies to joint inversion")

        targetclass = np.r_[
            [
                isinstance(dirpart, MultiTargetMisfits)
                for dirpart in self.inversion.directiveList.dList
            ]
        ]
        if ~np.any(targetclass):
            self.DMtarget = None
        else:
            self.targetclass = np.where(targetclass)[0][-1]
            self.DMtarget = self.inversion.directiveList.dList[
                self.targetclass
            ].DMtarget

        if self.verbose:
            print("Initial data misfit scales: ", self.dmisfit.multipliers)

    def endIter(self):

        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if np.any(self.dmlist < self.DMtarget):
            self.mode = 2
        else:
            self.mode = 1

        if self.opt.iter > 0 and self.opt.iter % self.update_rate == 0:

            if self.mode == 2:

                if np.all(np.r_[self.dmisfit.multipliers] > self.chimin) and np.all(
                    np.r_[self.dmisfit.multipliers] < self.chimax
                ):

                    indx = self.dmlist > self.DMtarget
                    if np.any(indx):
                        multipliers = self.warmingFactor * np.median(
                            self.DMtarget[~indx] / self.dmlist[~indx]
                        )
                        if np.sum(indx) == 1:
                            indx = np.where(indx)[0][0]
                        self.dmisfit.multipliers[indx] *= multipliers
                        self.dmisfit.multipliers /= np.sum(self.dmisfit.multipliers)

                        if self.verbose:
                            print("Updating scaling for data misfits by ", multipliers)
                            print("New scales:", self.dmisfit.multipliers)


class TargetMisfit(InversionDirective):
    """
    ... note:: Currently this target misfit is not set up for joint inversion.
    Check out MultiTargetMisfits
    """

    chifact = 1.0
    phi_d_star = None

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            # the factor of 0.5 is because we do phid = 0.5*||dpred - dobs||^2
            if self.phi_d_star is None:

                nD = 0
                for survey in self.survey:
                    nD += survey.nD

                self.phi_d_star = 0.5 * nD

            self._target = self.chifact * self.phi_d_star
        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def endIter(self):
        if self.invProb.phi_d < self.target:
            self.opt.stopNextIteration = True
            self.print_final_misfit()

    def print_final_misfit(self):
        if self.opt.print_type == "ubc":
            self.opt.print_target = (
                ">> Target misfit: %.1f (# of data) is achieved"
            ) % (self.target * self.invProb.opt.factor)


class MultiTargetMisfits(InversionDirective):

    WeightsInTarget = 0
    verbose = False
    # Chi factor for Geophsyical Data Misfit
    chifact = 1.0
    phi_d_star = None

    # Chifact for Clustering/Smallness
    TriggerSmall = True
    chiSmall = 1.0
    phi_ms_star = None

    # Tolerance for parameters difference with their priors
    TriggerTheta = False  # deactivated by default
    ToleranceTheta = 1.0
    distance_norm = np.inf

    AllStop = False
    DM = False  # geophysical fit condition
    CL = False  # petrophysical fit condition
    DP = False  # parameters difference with their priors condition

    def initialize(self):
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]

        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            (
                                isinstance(
                                    regpart, PGIwithNonlinearRelationshipsSmallness,
                                )
                                or isinstance(regpart, PGIsmallness)
                            ),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            if smallness[smallness[:, 2] == 1][:, :2].size == 0:
                warnings.warn(
                    "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)"
                )
                self.smallness = -1
                self.pgi_smallness = None

            else:
                self.smallness = smallness[smallness[:, 2] == 1][:, :2][0]
                self.pgi_smallness = self.invProb.reg.objfcts[
                    self.smallness[0]
                ].objfcts[self.smallness[1]]

                if self.debug:
                    print(
                        type(
                            self.invProb.reg.objfcts[self.smallness[0]].objfcts[
                                self.smallness[1]
                            ]
                        )
                    )

            self._regmode = 1

        else:
            smallness = np.r_[
                [
                    (
                        np.r_[
                            j,
                            (
                                isinstance(
                                    regpart, PGIwithNonlinearRelationshipsSmallness,
                                )
                                or isinstance(regpart, PGIsmallness)
                            ),
                        ]
                    )
                    for j, regpart in enumerate(self.invProb.reg.objfcts)
                ]
            ]
            if smallness[smallness[:, 1] == 1][:, :1].size == 0:
                if self.TriggerSmall:
                    warnings.warn(
                        "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)."
                    )
                    self.TriggerSmall = False
                self.smallness = -1
            else:
                self.smallness = smallness[smallness[:, 1] == 1][:, :1][0]
                self.pgi_smallness = self.invProb.reg.objfcts[self.smallness[0]]

                if self.debug:
                    print(type(self.invProb.reg.objfcts[self.smallness[0]]))

            self._regmode = 2

    @property
    def DMtarget(self):
        if getattr(self, "_DMtarget", None) is None:
            # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
            if self.phi_d_star is None:
                # Check if it is a ComboObjective
                if isinstance(self.dmisfit, ComboObjectiveFunction):
                    self.phi_d_star = np.r_[[0.5 * survey.nD for survey in self.survey]]
                else:
                    self.phi_d_star = np.r_[[0.5 * self.survey.nD]]

            self._DMtarget = self.chifact * self.phi_d_star
        return self._DMtarget

    @DMtarget.setter
    def DMtarget(self, val):
        self._DMtarget = val

    @property
    def CLtarget(self):
        if not getattr(self.pgi_smallness, "approx_eval", True):
            # if nonlinear prior, compute targer numerically at each GMM update
            samples, _ = self.pgi_smallness.gmm.sample(
                len(self.pgi_smallness.gmm.cell_volumes)
            )
            self.phi_ms_star = self.pgi_smallness(
                mkvc(samples), externalW=self.WeightsInTarget
            )

            self._CLtarget = self.chiSmall * self.phi_ms_star

        elif getattr(self, "_CLtarget", None) is None:
            # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
            if self.phi_ms_star is None:
                # Expected value is number of active cells * number of physical
                # properties
                self.phi_ms_star = 0.5 * len(self.invProb.model)

            self._CLtarget = self.chiSmall * self.phi_ms_star

        return self._CLtarget

    @property
    def CLnormalizedConstant(self):
        if ~self.WeightsInTarget:
            return 1.0
        elif np.any(self.smallness == -1):
            return np.sum(
                sp.csr_matrix.diagonal(self.invProb.reg.objfcts[0].W) ** 2.0
            ) / len(self.invProb.model)
        else:
            return np.sum(sp.csr_matrix.diagonal(self.pgi_smallness.W) ** 2.0) / len(
                self.invProb.model
            )

    @CLtarget.setter
    def CLtarget(self, val):
        self._CLtarget = val

    def phims(self):
        if np.any(self.smallness == -1):
            return self.invProb.reg.objfcts[0](self.invProb.model)
        else:
            return (
                self.pgi_smallness(self.invProb.model, externalW=self.WeightsInTarget,)
                / self.CLnormalizedConstant
            )

    def ThetaTarget(self):
        maxdiff = 0.0

        for i in range(self.invProb.reg.gmm.n_components):
            meandiff = np.linalg.norm(
                (self.invProb.reg.gmm.means_[i] - self.invProb.reg.gmmref.means_[i])
                / self.invProb.reg.gmmref.means_[i],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, meandiff)

            if (
                self.invProb.reg.gmm.covariance_type == "full"
                or self.invProb.reg.gmm.covariance_type == "spherical"
            ):
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_[i]
                        - self.invProb.reg.gmmref.covariances_[i]
                    )
                    / self.invProb.reg.gmmref.covariances_[i],
                    ord=self.distance_norm,
                )
            else:
                covdiff = np.linalg.norm(
                    (
                        self.invProb.reg.gmm.covariances_
                        - self.invProb.reg.gmmref.covariances_
                    )
                    / self.invProb.reg.gmmref.covariances_,
                    ord=self.distance_norm,
                )
            maxdiff = np.maximum(maxdiff, covdiff)

            pidiff = np.linalg.norm(
                [
                    (
                        self.invProb.reg.gmm.weights_[i]
                        - self.invProb.reg.gmmref.weights_[i]
                    )
                    / self.invProb.reg.gmmref.weights_[i]
                ],
                ord=self.distance_norm,
            )
            maxdiff = np.maximum(maxdiff, pidiff)

        return maxdiff

    def endIter(self):

        self.AllStop = False
        self.DM = False
        self.CL = True
        self.DP = True
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]
        self.targetlist = np.r_[
            [dm < tgt for dm, tgt in zip(self.dmlist, self.DMtarget)]
        ]

        if np.all(self.targetlist):
            self.DM = True

        if self.TriggerSmall and np.any(self.smallness != -1):
            if self.phims() > self.CLtarget:
                self.CL = False

        if self.TriggerTheta:
            if self.ThetaTarget() > self.ToleranceTheta:
                self.DP = False

        self.AllStop = self.DM and self.CL and self.DP
        if self.verbose:
            message = "geophys. misfits: " + "; ".join(
                map(
                    str,
                    [
                        "{0} (target {1} [{2}])".format(val, tgt, cond)
                        for val, tgt, cond in zip(
                            np.round(self.dmlist, 1),
                            np.round(self.DMtarget, 1),
                            self.targetlist,
                        )
                    ],
                )
            )
            if self.TriggerSmall:
                message += " | smallness misfit: {0:.1f} (target: {1:.1f} [{2}])".format(
                    self.phims(), self.CLtarget, self.CL
                )
            if self.TriggerTheta:
                message += " | GMM parameters within tolerance: {}".format(self.DP)
            print(message)

        if self.AllStop:
            self.opt.stopNextIteration = True
            if self.verbose:
                print("All targets have been reached")


class SaveEveryIteration(InversionDirective):
    """SaveEveryIteration

    This directive saves an array at each iteration. The default
    directory is the current directory and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    directory = properties.String("directory to save results in", default=".")

    name = properties.String(
        "root of the filename to be saved", default="InversionModel"
    )

    @properties.validator("directory")
    def _ensure_abspath(self, change):
        val = change["value"]
        fullpath = os.path.abspath(os.path.expanduser(val))

        if not os.path.isdir(fullpath):
            os.mkdir(fullpath)

    @property
    def fileName(self):
        if getattr(self, "_fileName", None) is None:
            from datetime import datetime

            self._fileName = "{0!s}-{1!s}".format(
                self.name, datetime.now().strftime("%Y-%m-%d-%H-%M")
            )
        return self._fileName


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration

    This directive saves the model as a numpy array at each iteration. The
    default directory is the current directoy and the models are saved as
    ``InversionModel-YYYY-MM-DD-HH-MM-iter.npy``
    """

    def initialize(self):
        print(
            "SimPEG.SaveModelEveryIteration will save your models as: "
            "'{0!s}###-{1!s}.npy'".format(self.directory + os.path.sep, self.fileName)
        )

    def endIter(self):
        np.save(
            "{0!s}{1:03d}-{2!s}".format(
                self.directory + os.path.sep, self.opt.iter, self.fileName
            ),
            self.opt.xc,
        )


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveOutputEveryIteration"""

    header = None
    save_txt = True
    beta = None
    phi_d = None
    phi_m = None
    phi = None

    def initialize(self):
        if self.save_txt is True:
            print(
                "SimPEG.SaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            self.header = "  #     beta     phi_d     phi_m      phi       cg_iter\n"
            f.write(self.header)
            f.close()

        # Create a list of each

        self.beta = []
        self.phi_d = []
        self.phi_m = []
        self.phi = []

    def endIter(self):

        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0

        if getattr(self.reg.objfcts[0], "objfcts", None) is not None:
            for reg in self.reg.objfcts:
                phi_s += reg.objfcts[0](self.invProb.model) * reg.alpha_s
                phi_x += reg.objfcts[1](self.invProb.model) * reg.alpha_x

                if reg.regmesh.dim == 2:
                    phi_y += reg.objfcts[2](self.invProb.model) * reg.alpha_y
                elif reg.regmesh.dim == 3:
                    phi_y += reg.objfcts[2](self.invProb.model) * reg.alpha_y
                    phi_z += reg.objfcts[3](self.invProb.model) * reg.alpha_z
        elif getattr(self.reg.objfcts[0], "objfcts", None) is None:
            phi_s += self.reg.objfcts[0](self.invProb.model) * self.reg.alpha_s
            phi_x += self.reg.objfcts[1](self.invProb.model) * self.reg.alpha_x

            if self.reg.regmesh.dim == 2:
                phi_y += self.reg.objfcts[2](self.invProb.model) * self.reg.alpha_y
            elif self.reg.regmesh.dim == 3:
                phi_y += self.reg.objfcts[2](self.invProb.model) * self.reg.alpha_y
                phi_z += self.reg.objfcts[3](self.invProb.model) * self.reg.alpha_z

        self.beta.append(self.invProb.beta)
        self.phi_d.append(self.invProb.phi_d)
        self.phi_m.append(self.invProb.phi_m)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            f.write(
                " {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:3d}\n".format(
                    self.opt.iter,
                    self.beta[self.opt.iter - 1],
                    self.phi_d[self.opt.iter - 1],
                    self.phi_m[self.opt.iter - 1],
                    self.phi[self.opt.iter - 1],
                    self.opt.cg_count,
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.beta = results[:, 1]
        self.phi_d = results[:, 2]
        self.phi_m = results[:, 3]
        self.f = results[:, 4]

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

    def plot_misfit_curves(
        self,
        fname=None,
        dpi=300,
        plot_small_smooth=False,
        plot_phi_m=True,
        plot_small=False,
        plot_smooth=False,
    ):

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 2))
        ax = plt.subplot(111)
        ax_1 = ax.twinx()
        ax.semilogy(
            np.arange(len(self.phi_d)), self.phi_d, "k-", lw=2, label="$\phi_d$"
        )

        if plot_phi_m:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m, "r", lw=2, label="$\phi_m$"
            )

        if plot_small_smooth or plot_small:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_small, "ro", label="small"
            )
        if plot_small_smooth or plot_smooth:
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_x, "rx", label="smooth_x"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_y, "rx", label="smooth_y"
            )
            ax_1.semilogy(
                np.arange(len(self.phi_d)), self.phi_m_smooth_z, "rx", label="smooth_z"
            )

        ax.legend(loc=1)
        ax_1.legend(loc=2)

        ax.plot(
            np.r_[ax.get_xlim()[0], ax.get_xlim()[1]],
            np.ones(2) * self.target_misfit,
            "k:",
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$\phi_d$")
        ax_1.set_ylabel("$\phi_m$", color="r")
        ax_1.tick_params(axis="y", which="both", colors="red")

        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def plot_tikhonov_curves(self, fname=None, dpi=200):

        self.target_misfit = self.invProb.dmisfit.simulation.survey.nD / 2.0
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

        fig = plt.figure(figsize=(5, 8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        ax1.plot(self.beta, self.phi_d, "k-", lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel("$\\beta$", fontsize=14)
        ax1.set_ylabel("$\phi_d$", fontsize=14)

        ax2.plot(self.beta, self.phi_m, "k-", lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel("$\\beta$", fontsize=14)
        ax2.set_ylabel("$\phi_m$", fontsize=14)

        ax3.plot(self.phi_m, self.phi_d, "k-", lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel("$\phi_m$", fontsize=14)
        ax3.set_ylabel("$\phi_d$", fontsize=14)

        if self.i_target is not None:
            ax1.plot(self.beta[self.i_target], self.phi_d[self.i_target], "k*", ms=10)
            ax2.plot(self.beta[self.i_target], self.phi_m[self.i_target], "k*", ms=10)
            ax3.plot(self.phi_m[self.i_target], self.phi_d[self.i_target], "k*", ms=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """
    Saves inversion parameters at every iteration.
    """

    # Initialize the output dict
    outDict = None
    saveOnDisk = False

    def initialize(self):
        self.outDict = {}
        if self.saveOnDisk:
            print(
                "SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(
                    self.fileName
                )
            )

    def endIter(self):

        # regCombo = ["phi_ms", "phi_msx"]

        # if self.simulation[0].mesh.dim >= 2:
        #     regCombo += ["phi_msy"]

        # if self.simulation[0].mesh.dim == 3:
        #     regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = {}

        # Save the data.
        iterDict["iter"] = self.opt.iter
        iterDict["beta"] = self.invProb.beta
        iterDict["phi_d"] = self.invProb.phi_d
        iterDict["phi_m"] = self.invProb.phi_m

        # for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
        #     iterDict[label] = fcts(self.invProb.model)

        iterDict["f"] = self.opt.f
        iterDict["m"] = self.invProb.model
        iterDict["dpred"] = self.invProb.dpred

        if hasattr(self.reg.objfcts[0], "eps_p") is True:
            iterDict["eps_p"] = self.reg.objfcts[0].eps_p
            iterDict["eps_q"] = self.reg.objfcts[0].eps_q

        if hasattr(self.reg.objfcts[0], "norms") is True:
            iterDict["lps"] = self.reg.objfcts[0].norms[0][0]
            iterDict["lpx"] = self.reg.objfcts[0].norms[0][1]

        # Save the file as a npz
        if self.saveOnDisk:

            np.savez("{:03d}-{:s}".format(self.opt.iter, self.fileName), iterDict)

        self.outDict[self.opt.iter] = iterDict


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
    max_irls_iterations = properties.Integer("maximum irls iterations", default=20)
    iterStart = 0
    sphericalDomain = False

    # Beta schedule
    update_beta = properties.Bool("Update beta", default=True)
    beta_search = properties.Bool("Do a beta search", default=False)
    coolingFactor = properties.Float("Cooling factor", default=2.0)
    coolingRate = properties.Integer("Cooling rate", default=1)
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

    maxIRLSiters = deprecate_property(
        max_irls_iterations,
        "maxIRLSiters",
        new_name="max_irls_iterations",
        removal_version="0.16.0",
        future_warn=True,
    )
    updateBeta = deprecate_property(
        update_beta,
        "updateBeta",
        new_name="update_beta",
        removal_version="0.16.0",
        future_warn=True,
    )
    betaSearch = deprecate_property(
        beta_search,
        "betaSearch",
        new_name="beta_search",
        removal_version="0.16.0",
        future_warn=True,
    )

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
                self.norms.append(reg.norms)
                reg.norms = np.c_[2.0, 2.0, 2.0, 2.0]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
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

        phim_new = 0
        for reg in self.reg.objfcts:
            for comp, multipier in zip(reg.objfcts, reg.multipliers):
                if multipier > 0:
                    phim_new += np.sum(
                        comp.f_m ** 2.0
                        / (comp.f_m ** 2.0 + comp.epsilon ** 2.0)
                        ** (1 - comp.norm / 2.0)
                    )

        # Update the model used by the regularization
        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.start, self.mode == 1]):
            self.startIRLS()

        # Only update after GN iterations
        if np.all(
            [(self.opt.iter - self.iterStart) % self.minGNiter == 0, self.mode != 1]
        ):

            if self.fix_Jmatrix:
                print(">> Fix Jmatrix")
                self.invProb.dmisfit.simulation.fix_Jmatrix = True
            # Check for maximum number of IRLS cycles
            if self.irls_iteration == self.max_irls_iterations:
                if not self.silent:
                    print(
                        "Reach maximum number of IRLS cycles:"
                        + " {0:d}".format(self.max_irls_iterations)
                    )

                self.opt.stopNextIteration = True
                return

            # Print to screen
            for reg in self.reg.objfcts:

                if reg.eps_p > self.floorEps_p and self.coolEps_p:
                    reg.eps_p /= self.coolEpsFact
                    # print('Eps_p: ' + str(reg.eps_p))
                if reg.eps_q > self.floorEps_q and self.coolEps_q:
                    reg.eps_q /= self.coolEpsFact
                    # print('Eps_q: ' + str(reg.eps_q))

            # Remember the value of the norm from previous R matrices
            # self.f_old = self.reg(self.invProb.model)

            self.irls_iteration += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:

                # If comboObj, go down one more level
                for comp in reg.objfcts:
                    comp.stashedR = None

            for dmis in self.dmisfit.objfcts:
                if getattr(dmis, "stashedR", None) is not None:
                    dmis.stashedR = None

            # Compute new model objective function value
            f_change = np.abs(self.f_old - phim_new) / (self.f_old + 1e-12)

            # Check if the function has changed enough
            if np.all(
                [
                    f_change < self.f_min_change,
                    self.irls_iteration > 1,
                    np.abs(1.0 - self.invProb.phi_d / self.target) < self.beta_tol,
                ]
            ):

                print("Minimum decrease in regularization." + "End of IRLS")
                self.opt.stopNextIteration = True
                return

            self.f_old = phim_new

            self.update_beta = True
            self.invProb.phi_m_last = self.reg(self.invProb.model)

    def startIRLS(self):
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

        # Either use the supplied epsilon, or fix base on distribution of
        # model values
        for reg in self.reg.objfcts:

            if getattr(reg, "eps_p", None) is None:

                reg.eps_p = np.percentile(
                    np.abs(reg.mapping * reg._delta_m(self.invProb.model)), self.prctile
                )

            if getattr(reg, "eps_q", None) is None:

                reg.eps_q = np.percentile(
                    np.abs(reg.mapping * reg._delta_m(self.invProb.model)), self.prctile
                )

        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms in zip(self.reg.objfcts, self.norms):
            reg.norms = norms

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        # Print to screen
        for reg in self.reg.objfcts:
            if not self.silent:
                print("eps_p: " + str(reg.eps_p) + " eps_q: " + str(reg.eps_q))

    def angleScale(self):
        """
        Update the scales used by regularization for the
        different block of models
        """
        # Currently implemented for MVI-S only
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            eps_p = reg.epsilon
            f_m = abs(reg.f_m)
            max_p += [np.max(f_m)]

        max_p = np.asarray(max_p).max()

        max_s = [np.pi, np.pi]

        for obj, var in zip(self.reg.objfcts[1:3], max_s):
            obj.scales = np.ones(obj.scales.shape) * max_p / var

    def validate(self, directiveList):
        # check if a linear preconditioner is in the list, if not warn else
        # assert that it is listed after the IRLS directive
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
                "directives list"
            )
        return True


class UpdatePreconditioner(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """

    update_every_iteration = True  #: Update every iterations if False

    def initialize(self):

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if regularization has a projection
            rdg = reg.deriv2(m)
            if not isinstance(rdg, Zero):
                regDiag += rdg.diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):

            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))

        self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.update_every_iteration is False:
            return

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)
        m = self.invProb.model

        for reg in self.reg.objfcts:
            # Check if he has wire
            regDiag += reg.deriv2(m).diagonal()

        JtJdiag = np.zeros_like(self.invProb.model)
        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):

            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )
                JtJdiag += np.sum(np.power((dmisfit.W * sim.getJ(m)), 2), axis=0)
            else:
                JtJdiag += sim.getJtJdiag(m, W=dmisfit.W)

        diagA = JtJdiag + self.invProb.beta * regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.0
        PC = sdiag((diagA))
        self.opt.approxHinv = PC


class Update_Wj(InversionDirective):
    """
    Create approx-sensitivity base weighting using the probing method
    """

    k = None  # Number of probing cycles
    itr = None  # Iteration number to update Wj, or always update if None

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


class UpdateSensitivityWeights(InversionDirective):
    """
    Directive to take care of re-weighting
    the non-linear problems. Assumes that the map of the regularization
    function is either Wires or Identity.
    Good for any problem where J is formed explicitly.
    """

    everyIter = True
    threshold = 1e-12
    normalization: bool = True

    def initialize(self):
        """
        Calculate and update sensitivity
        for optimization and regularization
        """
        for reg in self.reg.objfcts:
            if not isinstance(getattr(reg, "mapping"), (IdentityMap, Wires)):
                raise TypeError(
                    f"Mapping for the regularization must be of type {IdentityMap} or {Wires}. "
                    + f"Input mapping of type {type(reg.mapping)}."
                )

        self.update()

    def endIter(self):
        """
        Update inverse problem
        """
        if self.everyIter:
            self.update()

    def update(self):
        """
        Compute explicitly the main diagonal of JtJ

        """
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

        # Normalize and threshold weights
        wr = np.zeros_like(self.invProb.model)
        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                wr += reg.mapping.deriv(self.invProb.model).T * (
                    (reg.mapping * jtj_diag) / reg.objfcts[0].regmesh.vol ** 2.0
                )
        wr /= wr.max()
        wr += self.threshold
        wr **= 0.5
        for reg in self.reg.objfcts:
            if not isinstance(reg, BaseSimilarityMeasure):
                reg.cell_weights = reg.mapping * wr

    def validate(self, directiveList):
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)

        beta_estimator_ind = [isinstance(d, BetaEstimate_ByEig) for d in dList]

        lin_precond_ind = [isinstance(d, UpdatePreconditioner) for d in dList]

        if any(beta_estimator_ind):
            assert beta_estimator_ind.index(True) > self_ind, (
                "The directive 'BetaEstimate_ByEig' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        if any(lin_precond_ind):
            assert lin_precond_ind.index(True) > self_ind, (
                "The directive 'UpdatePreconditioner' must be after UpdateSensitivityWeights "
                "in the directiveList"
            )

        return True


class ProjectSphericalBounds(InversionDirective):
    """
    Trick for spherical coordinate system.
    Project \theta and \phi angles back to [-\pi,\pi] using
    back and forth conversion.
    spherical->cartesian->spherical
    """

    def initialize(self):

        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        nC = int(len(x) / 3)

        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        nC = int(len(x) / 3)

        # Convert to cartesian than back to avoid over rotation
        xyz = spherical2cartesian(x.reshape((nC, 3), order="F"))
        m = cartesian2spherical(xyz.reshape((nC, 3), order="F"))

        self.invProb.model = m

        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last

        for sim in self.simulation:
            sim.model = m

        self.opt.xc = m
