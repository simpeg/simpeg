import numpy as np
import warnings
import scipy.sparse as sp

from ..objective_function import ComboObjectiveFunction
from ..regularization import (
    PGIsmallness,
    PGIwithNonlinearRelationshipsSmallness,
)
from ..utils import mkvc, sdiag, validate_integer, Zero
from ..utils.code_utils import (
    validate_type,
    validate_float,
    validate_ndarray_with_shape,
)

from .base import InversionDirective


class TargetMisfit(InversionDirective):
    """
    ... note:: Currently this target misfit is not set up for joint inversion.
    Check out MultiTargetMisfits
    """

    def __init__(self, target=None, phi_d_star=None, chifact=1.0, **kwargs):
        super().__init__(**kwargs)
        self.chifact = chifact
        self.phi_d_star = phi_d_star
        if phi_d_star is not None and target is not None:
            raise AttributeError("Attempted to set both target and phi_d_star.")
        if target is not None:
            self.target = target

    @property
    def target(self):
        """The target value for the data misfit

        Returns
        -------
        float
        """
        if getattr(self, "_target", None) is None:
            self._target = self.chifact * self.phi_d_star
        return self._target

    @target.setter
    def target(self, val):
        self._target = validate_float("target", val, min_val=0.0, inclusive_min=False)

    @property
    def chifact(self):
        """The a multiplier for the target data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_float(
            "chifact", value, min_val=0.0, inclusive_min=False
        )
        self._target = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # the factor of 0.5 is because we do phid = 0.5*||dpred - dobs||^2
        if self._phi_d_star is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD
            self._phi_d_star = 0.5 * nD
        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # the factor of 0.5 is because we do phid = 0.5*||dpred - dobs||^2
        if value is not None:
            value = validate_float(
                "phi_d_star", value, min_val=0.0, inclusive_min=False
            )
        self._phi_d_star = value
        self._target = None

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
    def __init__(
        self,
        WeightsInTarget=False,
        chifact=1.0,
        phi_d_star=None,
        TriggerSmall=True,
        chiSmall=1.0,
        phi_ms_star=None,
        TriggerTheta=False,
        ToleranceTheta=1.0,
        distance_norm=np.inf,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.WeightsInTarget = WeightsInTarget
        # Chi factor for Geophsyical Data Misfit
        self.chifact = chifact
        self.phi_d_star = phi_d_star

        # Chifact for Clustering/Smallness
        self.TriggerSmall = TriggerSmall
        self.chiSmall = chiSmall
        self.phi_ms_star = phi_ms_star

        # Tolerance for parameters difference with their priors
        self.TriggerTheta = TriggerTheta  # deactivated by default
        self.ToleranceTheta = ToleranceTheta
        self.distance_norm = distance_norm

        self._DM = False
        self._CL = False
        self._DP = False

    @property
    def WeightsInTarget(self):
        """Whether to account for weights in the petrophysical misfit.

        Returns
        -------
        bool
        """
        return self._WeightsInTarget

    @WeightsInTarget.setter
    def WeightsInTarget(self, value):
        self._WeightsInTarget = validate_type("WeightsInTarget", value, bool)

    @property
    def chifact(self):
        """The a multiplier for the target Geophysical data misfit value.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        numpy.ndarray
        """
        return self._chifact

    @chifact.setter
    def chifact(self, value):
        self._chifact = validate_ndarray_with_shape("chifact", value, shape=("*",))
        self._DMtarget = None

    @property
    def phi_d_star(self):
        """The target phi_d value for the Geophysical data misfit.

        The target value is `chifact` times `phi_d_star`

        Returns
        -------
        float
        """
        # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
        if getattr(self, "_phi_d_star", None) is None:
            # Check if it is a ComboObjective
            if isinstance(self.dmisfit, ComboObjectiveFunction):
                value = np.r_[[0.5 * survey.nD for survey in self.survey]]
            else:
                value = np.r_[[0.5 * self.survey.nD]]
            self._phi_d_star = value
            self._DMtarget = None

        return self._phi_d_star

    @phi_d_star.setter
    def phi_d_star(self, value):
        # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
        if value is not None:
            value = validate_ndarray_with_shape("phi_d_star", value, shape=("*",))
        self._phi_d_star = value
        self._DMtarget = None

    @property
    def chiSmall(self):
        """The a multiplier for the target petrophysical misfit value.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._chiSmall

    @chiSmall.setter
    def chiSmall(self, value):
        self._chiSmall = validate_float("chiSmall", value)
        self._CLtarget = None

    @property
    def phi_ms_star(self):
        """The target value for the petrophysical data misfit.

        The target value is `chiSmall` times `phi_ms_star`

        Returns
        -------
        float
        """
        return self._phi_ms_star

    @phi_ms_star.setter
    def phi_ms_star(self, value):
        if value is not None:
            value = validate_float("phi_ms_star", value)
        self._phi_ms_star = value
        self._CLtarget = None

    @property
    def TriggerSmall(self):
        """Whether to trigger the smallness misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerSmall

    @TriggerSmall.setter
    def TriggerSmall(self, value):
        self._TriggerSmall = validate_type("TriggerSmall", value, bool)

    @property
    def TriggerTheta(self):
        """Whether to trigger the GMM misfit test.

        Returns
        -------
        bool
        """
        return self._TriggerTheta

    @TriggerTheta.setter
    def TriggerTheta(self, value):
        self._TriggerTheta = validate_type("TriggerTheta", value, bool)

    @property
    def ToleranceTheta(self):
        """Target value for the GMM misfit.

        Returns
        -------
        float
        """
        return self._ToleranceTheta

    @ToleranceTheta.setter
    def ToleranceTheta(self, value):
        self._ToleranceTheta = validate_float("ToleranceTheta", value, min_val=0.0)

    @property
    def distance_norm(self):
        """Distance norm to use for GMM misfit measure.

        Returns
        -------
        float
        """
        return self._distance_norm

    @distance_norm.setter
    def distance_norm(self, value):
        self._distance_norm = validate_float("distance_norm", value, min_val=0.0)

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
                                    regpart,
                                    PGIwithNonlinearRelationshipsSmallness,
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
                    "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag)",
                    stacklevel=2,
                )
                self.smallness = -1
                self.pgi_smallness = None

            else:
                self.smallness = smallness[smallness[:, 2] == 1][:, :2][0]
                self.pgi_smallness = self.invProb.reg.objfcts[
                    self.smallness[0]
                ].objfcts[self.smallness[1]]

                if self.verbose:
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
                                    regpart,
                                    PGIwithNonlinearRelationshipsSmallness,
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
                        "There is no PGI regularization. Smallness target is turned off (TriggerSmall flag).",
                        stacklevel=2,
                    )
                    self.TriggerSmall = False
                self.smallness = -1
            else:
                self.smallness = smallness[smallness[:, 1] == 1][:, :1][0]
                self.pgi_smallness = self.invProb.reg.objfcts[self.smallness[0]]

                if self.verbose:
                    print(type(self.invProb.reg.objfcts[self.smallness[0]]))

            self._regmode = 2

    @property
    def DM(self):
        """Whether the geophysical data misfit target was satisfied.

        Returns
        -------
        bool
        """
        return self._DM

    @property
    def CL(self):
        """Whether the petrophysical misfit target was satisified.

        Returns
        -------
        bool
        """
        return self._CL

    @property
    def DP(self):
        """Whether the GMM misfit was below the threshold.

        Returns
        -------
        bool
        """
        return self._DP

    @property
    def AllStop(self):
        """Whether all target misfit values have been met.

        Returns
        -------
        bool
        """

        return self.DM and self.CL and self.DP

    @property
    def DMtarget(self):
        if getattr(self, "_DMtarget", None) is None:
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
                self.pgi_smallness(
                    self.invProb.model, external_weights=self.WeightsInTarget
                )
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
        self._DM = False
        self._CL = True
        self._DP = True
        self.dmlist = np.r_[[dmis(self.invProb.model) for dmis in self.dmisfit.objfcts]]
        self.targetlist = np.r_[
            [dm < tgt for dm, tgt in zip(self.dmlist, self.DMtarget)]
        ]

        if np.all(self.targetlist):
            self._DM = True

        if self.TriggerSmall and np.any(self.smallness != -1):
            if self.phims() > self.CLtarget:
                self._CL = False

        if self.TriggerTheta:
            if self.ThetaTarget() > self.ToleranceTheta:
                self._DP = False

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
                message += (
                    " | smallness misfit: {0:.1f} (target: {1:.1f} [{2}])".format(
                        self.phims(), self.CLtarget, self.CL
                    )
                )
            if self.TriggerTheta:
                message += " | GMM parameters within tolerance: {}".format(self.DP)
            print(message)

        if self.AllStop:
            self.opt.stopNextIteration = True
            if self.verbose:
                print("All targets have been reached")


class UpdatePreconditioner(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """

    def __init__(self, update_every_iteration=True, **kwargs):
        super().__init__(**kwargs)
        self.update_every_iteration = update_every_iteration

    @property
    def update_every_iteration(self):
        """Whether to update the preconditioner at every iteration.

        Returns
        -------
        bool
        """
        return self._update_every_iteration

    @update_every_iteration.setter
    def update_every_iteration(self, value):
        self._update_every_iteration = validate_type(
            "update_every_iteration", value, bool
        )

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


class MovingAndMultiTargetStopping(InversionDirective):
    r"""
    Directive for setting stopping criteria for a joint inversion.
    Ensures both that all target misfits are met and there is a small change in the
    model. Computes the percentage change of the current model from the previous model.

    ..math::
        \frac {\| \mathbf{m_i} - \mathbf{m_{i-1}} \|} {\| \mathbf{m_{i-1}} \|}
    """

    tol = 1e-5
    beta_tol = 1e-1
    chifact_target = 1.0

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = []
            for survey in self.survey:
                nD += [survey.nD]
            nD = np.array(nD)

            self._target = nD * 0.5 * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def endIter(self):
        for phi_d, target in zip(self.invProb.phi_d_list, self.target):
            if np.abs(1.0 - phi_d / target) >= self.beta_tol:
                return
        if (
            np.linalg.norm(self.opt.xc - self.opt.x_last)
            / np.linalg.norm(self.opt.x_last)
            > self.tol
        ):
            return

        print(
            "stopping criteria met: ",
            np.linalg.norm(self.opt.xc - self.opt.x_last)
            / np.linalg.norm(self.opt.x_last),
        )
        self.opt.stopNextIteration = True


class BetaSchedule(InversionDirective):
    """Reduce trade-off parameter (beta) at successive iterations using a cooling schedule.

    Updates the **beta** property in the associated :class:`SimPEG.inverse_problem.BaseInvProblem`
    while the inversion is running.
    For linear least-squares problems, the optimization problem can be solved in a
    single step and the cooling rate can be set to *1*. For non-linear optimization
    problems, multiple steps are required obtain the minimizer for a fixed trade-off
    parameter. In this case, the cooling rate should be larger than 1.

    Parameters
    ----------
    coolingFactor : float
        The factor by which the trade-off parameter is decreased when updated.
        The preexisting value of the trade-off parameter is divided by the cooling factor.
    coolingRate : int
        Sets the number of successive iterations before the trade-off parameter is reduced.
        Use *1* for linear least-squares optimization problems. Use *2* for weakly non-linear
        optimization problems. Use *3* for general non-linear optimization problems.

    """

    def __init__(self, coolingFactor=8.0, coolingRate=3, **kwargs):
        super().__init__(**kwargs)
        self.coolingFactor = coolingFactor
        self.coolingRate = coolingRate

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

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.verbose:
                print(
                    "BetaSchedule is cooling Beta. Iteration: {0:d}".format(
                        self.opt.iter
                    )
                )
            self.invProb.beta /= self.coolingFactor


class PairedBetaSchedule(InversionDirective):
    """
    Directive for beta cooling schedule to determine the tradeoff
    parameters when using paired data misfits and regularizations for a joint inversion.
    """

    chifact_target = 1.0
    beta_tol = 1e-1
    update_beta = True
    cooling_rate = 1
    cooling_factor = 2
    dmis_met = False

    @property
    def target(self):
        if getattr(self, "_target", None) is None:
            nD = np.array([survey.nD for survey in self.survey])

            self._target = nD * 0.5 * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):
        self.dmis_met = np.zeros_like(self.invProb.betas, dtype=bool)

    def endIter(self):
        # Check if target misfit has been reached, if so, set dmis_met to True
        for i, phi_d in enumerate(self.invProb.phi_d_list):
            self.dmis_met[i] = phi_d < self.target[i]

        # check separately if misfits are within the tolerance,
        # otherwise, scale beta individually
        for i, phi_d in enumerate(self.invProb.phi_d_list):
            if self.opt.iter > 0 and self.opt.iter % self.cooling_rate == 0:
                target = self.target[i]
                ratio = phi_d / target
                if self.update_beta and ratio <= (1.0 + self.beta_tol):
                    if ratio <= 1:
                        ratio = np.maximum(0.75, ratio)
                    else:
                        ratio = np.minimum(1.5, ratio)

                    self.invProb.betas[i] /= ratio
                elif ratio > 1.0:
                    self.invProb.betas[i] /= self.cooling_factor

        self.reg.multipliers[:-1] = self.invProb.betas
