from __future__ import print_function

import properties
import numpy as np

norm = np.linalg.norm
import matplotlib.pyplot as plt
import warnings
import os
from ..data_misfit import BaseDataMisfit
from ..objective_function import ComboObjectiveFunction
from ..maps import SphericalSystem, ComboMap
from ..regularization import BaseComboRegularization, BaseRegularization, BaseCoupling
from ..utils import (
    mkvc,
    setKwargs,
    sdiag,
    diagEst,
    spherical2cartesian,
    cartesian2spherical,
    eigenvalue_by_power_iteration,  # change on Dec 9, 2020
)
from ..utils.code_utils import deprecate_property
from .. import optimization
from .directives import InversionDirective, SaveEveryIteration, UpdatePreconditioner

###############################################################################
#                                                                             #
#              Directives of joint inversion                                  #
#                                                                             #
###############################################################################

IterationPrinters = optimization.IterationPrinters
StoppingCriteria = optimization.StoppingCriteria


class CrossGradientInversionDirective(InversionDirective):
    """
        Directive for cross gradient inversions. Sets Printers and StoppingCriteria.
        Methods assume we are working with two models.
    """

    class JointInversionPrinters(IterationPrinters):
        betas = {
            "title": "betas",
            "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.betas],
            "width": 26,
            "format": "%s",
        }
        lambd = {
            "title": "lambda",
            "value": lambda M: M.parent.lambd,
            "width": 10,
            "format": "%1.2e",
        }
        phi_d_joint = {
            "title": "phi_d",
            "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.phi_d_joint],
            "width": 26,
            "format": "%s",
        }
        phi_m_joint = {
            "title": "phi_m",
            "value": lambda M: ["{:.2e}".format(elem) for elem in M.parent.phi_m_joint],
            "width": 26,
            "format": "%s",
        }
        phi_c = {
            "title": "phi_c",
            "value": lambda M: M.parent.phi_c,
            "width": 10,
            "format": "%1.2e",
        }
        iterationCG = {
            "title": "iterCG",
            "value": lambda M: M.cg_count,
            "width": 10,
            "format": "%3d",
        }

    printers = [
        IterationPrinters.iteration,
        JointInversionPrinters.betas,
        JointInversionPrinters.lambd,
        IterationPrinters.f,
        JointInversionPrinters.phi_d_joint,
        JointInversionPrinters.phi_m_joint,
        JointInversionPrinters.phi_c,
        JointInversionPrinters.iterationCG,
    ]

    def initialize(self):
        ### define relevant attributes
        self.betas = self.reg.multipliers[:-1]
        self.lambd = self.reg.multipliers[-1]
        self.phi_d_joint = []
        self.phi_m_joint = []
        self.phi_c = 0.0

        ### pass attributes to invProb
        self.invProb.betas = self.betas
        self.invProb.num_models = len(self.betas)
        self.invProb.lambd = self.lambd
        self.invProb.phi_d_joint = self.phi_d_joint
        self.invProb.phi_m_joint = self.phi_m_joint
        self.invProb.phi_c = self.phi_c

        self.opt.printers = self.printers
        self.opt.stoppers = [StoppingCriteria.iteration]

    def validate(self, directiveList):
        # check that this directive is first in the DirectiveList
        dList = directiveList.dList
        self_ind = dList.index(self)
        assert self_ind == 0, "The Joint_InversionDirective must be first."

        return True

    def endIter(self):
        ### compute attribute values
        phi_d = []
        for dmis in self.dmisfit.objfcts:
            phi_d.append(dmis(self.opt.xc))

        phi_m = []
        for reg in self.reg.objfcts:
            phi_m.append(reg(self.opt.xc))

        ### pass attributes values to invProb
        ### Assume last reg.objfct is the coupling
        self.invProb.phi_d_joint = phi_d
        self.invProb.phi_m_joint = phi_m[:-1]
        self.invProb.phi_c = phi_m[-1]
        self.invProb.betas = self.reg.multipliers[:-1]
        self.invProb.lambd = self.reg.multipliers[-1]


class Joint_SaveOutputEveryIteration(SaveEveryIteration, InversionDirective):
    """
    SaveOutputEveryIteration for Joint Inversions.
    Saves information on the tradeoff parameters, data misfits, regularizations,
    coupling term, number of CG iterations, and value of cost function.
    """

    header = None
    save_txt = True
    betas = None
    phi_d = None
    phi_m = None
    phi_c = None
    phi = None

    def initialize(self):
        if self.save_txt is True:
            print(
                "SimPEG.SaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName + ".txt", "w")
            self.header = "  #          betas            lambda         joint_phi_d                joint_phi_m            phi_c       iterCG     phi    \n"
            f.write(self.header)
            f.close()

        # Create a list of each
        self.betas = []
        self.lambd = []
        self.phi_d = []
        self.phi_m = []
        self.phi = []
        self.phi_c = []

    def endIter(self):

        self.betas.append(["{:.2e}".format(elem) for elem in self.invProb.betas])
        self.phi_d.append(["{:.3e}".format(elem) for elem in self.invProb.phi_d_joint])
        self.phi_m.append(["{:.3e}".format(elem) for elem in self.invProb.phi_m_joint])
        self.lambd.append("{:.2e}".format(self.invProb.lambd))
        self.phi_c.append(self.invProb.phi_c)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            f.write(
                " {0:2d}  {1}  {2}  {3}  {4}  {5:1.4e}  {6:d}  {7:1.4e}\n".format(
                    self.opt.iter,
                    self.betas[self.opt.iter - 1],
                    self.lambd[self.opt.iter - 1],
                    self.phi_d[self.opt.iter - 1],
                    self.phi_m[self.opt.iter - 1],
                    self.phi_c[self.opt.iter - 1],
                    self.opt.cg_count,
                    self.phi[self.opt.iter - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
        self.betas = results[:, 1]
        self.lambd = results[:, 2]
        self.phi_d = results[:, 3]
        self.phi_m = results[:, 4]
        self.phi_c = results[:, 5]
        self.f = results[:, 7]


class PairedBetaEstimate_ByEig(InversionDirective):
    """
    Estimate the trade-off parameter beta between pairs of data misfit(s) and the
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
                \\mathbf{x_1 = A x_0}
            Given this (very course) approximation of the eigenvector, we can
            use the *Rayleigh quotient* to approximate the largest eigenvalue.
            .. math::
                \\lambda_0 = \\frac{\\mathbf{x^\\top A x}}{\\mathbf{x^\\top x}}
            We will approximate the largest eigenvalue for both JtJ and WtW,
            and use some ratio of the quotient to estimate beta0.
            .. math::
                \\beta_0 = \\gamma \\frac{\\mathbf{x^\\top J^\\top J x}}{\\mathbf{x^\\top W^\\top W x}}
            :rtype: float
            :return: beta0
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.debug:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model
        dmis_eigenvalues = []
        reg_eigenvalues = []
        dmis_objs = self.dmisfit.objfcts
        reg_objs = [
            obj for obj in self.reg.objfcts if not isinstance(obj, BaseCoupling)
        ]
        if len(dmis_objs) != len(reg_objs):
            raise ValueError(
                f"There must be the same number of data misfit and regularizations. "
                f"Got {len(dmis_objs)} and {len(reg_objs)} respectively."
            )
        for dmis, reg in zip(dmis_objs, reg_objs):
            dmis_eigenvalues.append(
                eigenvalue_by_power_iteration(dmis, m, n_pw_iter=self.n_pw_iter,)
            )

            reg_eigenvalues.append(
                eigenvalue_by_power_iteration(reg, m, n_pw_iter=self.n_pw_iter,)
            )

        self.ratios = np.array(dmis_eigenvalues) / np.array(reg_eigenvalues)
        self.invProb.betas = self.beta0_ratio * self.ratios
        self.reg.multipliers[:-1] = self.invProb.betas


class Joint_BetaSchedule(InversionDirective):
    """
        Directive for beta cooling schedule to determine the tradeoff
        parameters of the joint inverse problem.
        We borrow some code from Update_IRLS.
    """

    chifact_target = 1.0
    beta_tol = 1e-1
    update_beta = True
    coolingRate = 1
    coolingFactor = 2
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

        self.betas = self.invProb.betas
        self.dmis_met = np.zeros_like(self.betas, dtype=int)
        self.dmis_met = self.dmis_met.astype(bool)

    def endIter(self):

        # Check if target misfit has been reached, if so, set dmis_met to True
        for i in range(self.invProb.num_models):
            if self.invProb.phi_d_joint[i] < self.target[i]:
                self.dmis_met[i] = True

        # check separately if misfits are within the tolerance,
        # otherwise, scale beta individually
        for i in range(self.invProb.num_models):
            if np.all(
                [
                    np.abs(1.0 - self.invProb.phi_d_joint[i] / self.target[i])
                    > self.beta_tol,
                    self.update_beta,
                    self.dmis_met[i],
                    self.opt.iter % self.coolingRate == 0,
                ]
            ):
                ratio = self.target[i] / self.invProb.phi_d_joint[i]
                if ratio > 1:
                    ratio = np.minimum(1.5, ratio)
                else:
                    ratio = np.maximum(0.75, ratio)

                self.invProb.betas[i] = self.invProb.betas[i] * ratio

            elif np.all(
                [self.opt.iter % self.coolingRate == 0, self.dmis_met[i] == False]
            ):
                self.invProb.betas[i] = self.invProb.betas[i] / self.coolingFactor

        self.reg.multipliers[:-1] = self.invProb.betas


class Joint_Stopping(InversionDirective):
    """
        Directive for setting Joint_StoppingCriteria.
        Computes the percentage change of the current model from the previous model.
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
        criteria_1 = (
            np.abs(1.0 - self.invProb.phi_d_joint[0] / self.target[0]) < self.beta_tol,
        )
        criteria_2 = (
            np.abs(1.0 - self.invProb.phi_d_joint[-1] / self.target[-1])
            < self.beta_tol,
        )
        criteria_3 = (
            norm(self.opt.xc - self.opt.x_last) / norm(self.opt.x_last) < self.tol
        )
        if np.all([criteria_1, criteria_2, criteria_3]):
            print(
                "stopping criteria met: ",
                norm(self.opt.xc - self.opt.x_last) / norm(self.opt.x_last),
            )
            self.opt.stopNextIteration = True


class Joint_UpdateSensitivityWeights(InversionDirective):
    """
    Directive to take care of re-weighting
    the non-linear magnetic problems.
    """

    mapping = None
    JtJdiag = None
    everyIter = True
    threshold = 1e-12
    switch = True

    def initialize(self):

        # Calculate and update sensitivity
        # for optimization and regularization
        self.update()

    def endIter(self):

        if self.everyIter:
            # Update inverse problem
            self.update()

    def update(self):

        # Get sum square of columns of J
        self.getJtJdiag()

        # Compute normalized weights
        self.wr = self.getWr()

        # Update the regularization
        self.updateReg()

    def getJtJdiag(self):
        """
            Compute explicitly the main diagonal of JtJ
            Good for any problem where J is formed explicitly
        """
        self.JtJdiag = []
        m = self.invProb.model

        for sim, dmisfit in zip(self.simulation, self.dmisfit.objfcts):

            if getattr(sim, "getJtJdiag", None) is None:
                assert getattr(sim, "getJ", None) is not None, (
                    "Simulation does not have a getJ attribute."
                    + "Cannot form the sensitivity explicitly"
                )

                self.JtJdiag += [
                    mkvc(np.sum((dmisfit.W * sim.getJ(m)) ** (2.0), axis=0))
                ]
            else:
                self.JtJdiag += [sim.getJtJdiag(m, W=dmisfit.W)]

        return self.JtJdiag

    def getWr(self):
        """
            Take the diagonal of JtJ and return
            a normalized sensitivty weighting vector
        """

        wr = np.zeros_like(self.invProb.model)
        if self.switch:
            for prob_JtJ, sim, dmisfit in zip(
                self.JtJdiag, self.simulation, self.dmisfit.objfcts
            ):

                wr += prob_JtJ + self.threshold

            wr = wr ** 0.5
            wr /= wr.max()
        else:
            wr += 1.0

        return wr

    def updateReg(self):
        """
            Update the cell weights with the approximated sensitivity
        """

        for reg in self.reg.objfcts[:-1]:
            reg.cell_weights = reg.mapping * (self.wr)

    def validate(self, directiveList):
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)

        beta_estimator_ind = [isinstance(d, Joint_BetaEstimate_ByEig) for d in dList]

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
