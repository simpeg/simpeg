from __future__ import print_function

import properties
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from . import Utils
from . import Regularization, DataMisfit, ObjectiveFunction
from . import Optimization
from . import Maps
from .Utils import mkvc
norm = np.linalg.norm
IterationPrinters = Optimization.IterationPrinters
StoppingCriteria = Optimization.StoppingCriteria


class InversionDirective(properties.HasProperties):
    """InversionDirective"""

    debug = False    #: Print debugging information
    _regPair = [
        Regularization.BaseComboRegularization,
        Regularization.BaseRegularization,
        ObjectiveFunction.ComboObjectiveFunction
    ]
    _dmisfitPair = [
        DataMisfit.BaseDataMisfit,
        ObjectiveFunction.ComboObjectiveFunction
    ]

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self, '_inversion', None)

    @inversion.setter
    def inversion(self, i):
        if getattr(self, '_inversion', None) is not None:
            warnings.warn(
                'InversionDirective {0!s} has switched to a new inversion.'
                .format(self.__class__.__name__)
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
        if getattr(self, '_reg', None) is None:
            self.reg = self.invProb.reg  # go through the setter
        return self._reg

    @reg.setter
    def reg(self, value):
        assert any([isinstance(value, regtype) for regtype in self._regPair]), (
            "Regularization must be in {}, not {}".format(
                self._regPair, type(value)
            )
        )

        if isinstance(value, Regularization.BaseComboRegularization):
            value = 1*value  # turn it into a combo objective function
        self._reg = value

    @property
    def dmisfit(self):
        if getattr(self, '_dmisfit', None) is None:
            self.dmisfit = self.invProb.dmisfit  # go through the setter
        return self._dmisfit

    @dmisfit.setter
    def dmisfit(self, value):

        assert any([
                isinstance(value, dmisfittype) for dmisfittype in
                self._dmisfitPair
        ]), "Regularization must be in {}, not {}".format(
                self._dmisfitPair, type(value)
        )

        if not isinstance(value, ObjectiveFunction.ComboObjectiveFunction):
            value = 1*value  # turn it into a combo objective function
        self._dmisfit = value

    @property
    def survey(self):
        """
           Assuming that dmisfit is always a ComboObjectiveFunction,
           return a list of surveys for each dmisfit [survey1, survey2, ... ]
        """
        return [objfcts.survey for objfcts in self.dmisfit.objfcts]

    @property
    def prob(self):
        """
           Assuming that dmisfit is always a ComboObjectiveFunction,
           return a list of problems for each dmisfit [prob1, prob2, ...]
        """
        return [objfcts.prob for objfcts in self.dmisfit.objfcts]

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass

    def validate(self, directiveList=None):
        return True


class DirectiveList(object):

    dList = None   #: The list of Directives

    def __init__(self, *directives, **kwargs):
        self.dList = []
        for d in directives:
            assert isinstance(d, InversionDirective), (
                'All directives must be InversionDirectives not {}'
                .format(type(d))
            )
            self.dList.append(d)
        Utils.setKwargs(self, **kwargs)

    @property
    def debug(self):
        return getattr(self, '_debug', False)

    @debug.setter
    def debug(self, value):
        for d in self.dList:
            d.debug = value
        self._debug = value

    @property
    def inversion(self):
        """This is the inversion of the InversionDirective instance."""
        return getattr(self, '_inversion', None)

    @inversion.setter
    def inversion(self, i):
        if self.inversion is i:
            return
        if getattr(self, '_inversion', None) is not None:
            warnings.warn(
                '{0!s} has switched to a new inversion.'
                .format(self.__class__.__name__)
            )
        for d in self.dList:
            d.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.dList is None:
            if self.debug:
                print('DirectiveList is None, no directives to call!')
            return

        directives = ['initialize', 'endIter', 'finish']
        assert ruleType in directives, (
            'Directive type must be in ["{0!s}"]'
            .format('", "'.join(directives))
        )
        for r in self.dList:
            getattr(r, ruleType)()

    def validate(self):
        [directive.validate(self) for directive in self.dList]
        return True


class BetaEstimate_ByEig(InversionDirective):
    """BetaEstimate"""

    beta0 = None       #: The initial Beta (regularization parameter)
    beta0_ratio = 1e2  #: estimateBeta0 is used with this ratio

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

        if self.debug:
            print('Calculating the beta0 parameter.')

        m = self.invProb.model
        f = self.invProb.getFields(m, store=True, deleteWarmstart=False)

        # Fix the seed for random vector for consistent result
        np.random.seed(1)
        x0 = np.random.rand(*m.shape)

        t, b = 0, 0
        i_count = 0
        for dmis, reg in zip(self.dmisfit.objfcts, self.reg.objfcts):
            # check if f is list
            if len(self.dmisfit.objfcts) > 1:
                t += x0.dot(dmis.deriv2(m, x0, f=f[i_count]))
            else:
                t += x0.dot(dmis.deriv2(m, x0, f=f))
            b += x0.dot(reg.deriv2(m, v=x0))
            i_count += 1

        self.beta0 = self.beta0_ratio*(t/b)
        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """BetaSchedule"""

    coolingFactor = 8.
    coolingRate = 3

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug:
                print(
                    'BetaSchedule is cooling Beta. Iteration: {0:d}'
                    .format(self.opt.iter)
                )
            self.invProb.beta /= self.coolingFactor


class TargetMisfit(InversionDirective):
    """
    ... note:: Currently the target misfit is not set up for joint inversions. Get `in touch <https://github.com/simpeg/simpeg/issues/new>`_ if you would like to help with the upgrade!
    """
    chifact = 1.
    phi_d_star = None

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
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


class SaveEveryIteration(InversionDirective):
    """SaveEveryIteration

    This directive saves an array at each iteration. The default
    direcroty is the current directoy and the models are saved as
    `InversionModel-YYYY-MM-DD-HH-MM-iter.npy`
    """

    directory = properties.String(
        "directory to save results in",
        default = "."
    )

    name = properties.String(
        "root of the filename to be saved",
        default="InversionModel"
    )

    @properties.validator('directory')
    def _ensure_abspath(self, change):
        val = change['value']
        fullpath = os.path.abspath(os.path.expanduser(val))

        if not os.path.isdir(fullpath):
            os.mkdir(fullpath)

    @property
    def fileName(self):
        if getattr(self, '_fileName', None) is None:
            from datetime import datetime
            self._fileName = '{0!s}-{1!s}'.format(
                self.name, datetime.now().strftime('%Y-%m-%d-%H-%M')
            )
        return self._fileName


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration

    This directive saves the model as a numpy array at each iteration. The
    default direcroty is the current directoy and the models are saved as
    `InversionModel-YYYY-MM-DD-HH-MM-iter.npy`
    """

    def initialize(self):
        print(
            "SimPEG.SaveModelEveryIteration will save your models as: "
            "'{0!s}###-{1!s}.npy'".format(
                self.directory + os.path.sep, self.fileName
            )
        )

    def endIter(self):
        np.save(
            '{0!s}{1:03d}-{2!s}'.format(
                self.directory + os.path.sep, self.opt.iter, self.fileName
            ), self.opt.xc
        )


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveOutputEveryIteration"""

    header = None
    save_txt = True
    beta = None
    phi_d = None
    phi_m = None
    phi_m_small = None
    phi_m_smooth_x = None
    phi_m_smooth_y = None
    phi_m_smooth_z = None
    phi = None

    def initialize(self):
        if self.save_txt is True:
            print(
                "SimPEG.SaveOutputEveryIteration will save your inversion "
                "progress as: '###-{0!s}.txt'".format(self.fileName)
            )
            f = open(self.fileName+'.txt', 'w')
            self.header = "  #     beta     phi_d     phi_m   phi_m_small     phi_m_smoomth_x     phi_m_smoomth_y     phi_m_smoomth_z      phi\n"
            f.write(self.header)
            f.close()

        # Create a list of each

        self.beta = []
        self.phi_d = []
        self.phi_m = []
        self.phi_m_small = []
        self.phi_m_smooth_x = []
        self.phi_m_smooth_y = []
        self.phi_m_smooth_z = []
        self.phi = []

    def endIter(self):

        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0
        for reg in self.reg.objfcts:
            phi_s += (
                reg.objfcts[0](self.invProb.model) * reg.alpha_s
            )
            phi_x += (
                reg.objfcts[1](self.invProb.model) * reg.alpha_x
            )

            if reg.regmesh.dim == 2:
                phi_y += (
                    reg.objfcts[2](self.invProb.model) * reg.alpha_y
                )
            elif reg.regmesh.dim == 3:
                phi_y += (
                    reg.objfcts[2](self.invProb.model) * reg.alpha_y
                )
                phi_z += (
                    reg.objfcts[3](self.invProb.model) * reg.alpha_z
                )

        self.beta.append(self.invProb.beta)
        self.phi_d.append(self.invProb.phi_d)
        self.phi_m.append(self.invProb.phi_m)
        self.phi_m_small.append(phi_s)
        self.phi_m_smooth_x.append(phi_x)
        self.phi_m_smooth_y.append(phi_y)
        self.phi_m_smooth_z.append(phi_z)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName+'.txt', 'a')
            f.write(
                ' {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:1.4e} '
                '{6:1.4e}  {7:1.4e}  {8:1.4e}\n'.format(
                    self.opt.iter,
                    self.beta[self.opt.iter-1],
                    self.phi_d[self.opt.iter-1],
                    self.phi_m[self.opt.iter-1],
                    self.phi_m_small[self.opt.iter-1],
                    self.phi_m_smooth_x[self.opt.iter-1],
                    self.phi_m_smooth_y[self.opt.iter-1],
                    self.phi_m_smooth_z[self.opt.iter-1],
                    self.phi[self.opt.iter-1]
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName+str(".txt"), comments="#")
        self.beta = results[:, 1]
        self.phi_d = results[:, 2]
        self.phi_m = results[:, 3]
        self.phi_m_small = results[:, 4]
        self.phi_m_smooth_x = results[:, 5]
        self.phi_m_smooth_y = results[:, 6]
        self.phi_m_smooth_z = results[:, 7]

        self.phi_m_smooth = (
                self.phi_m_smooth_x + self.phi_m_smooth_y + self.phi_m_smooth_z
                )

        self.f = results[:, 7]

        self.target_misfit = self.invProb.dmisfit.prob.survey.nD / 2.
        self.i_target = None

        if self.invProb.phi_d < self.target_misfit:
            i_target = 0
            while self.phi_d[i_target] > self.target_misfit:
                i_target += 1
            self.i_target = i_target

    def plot_misfit_curves(
        self, fname=None, dpi=300,
        plot_small_smooth=False,
        plot_phi_m=True,
        plot_small=False,
        plot_smooth=False
    ):

        self.target_misfit = self.invProb.dmisfit.prob.survey.nD / 2.
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
            np.arange(len(self.phi_d)),
            self.phi_d, 'k-', lw=2,
            label="$\phi_d$"
        )

        if plot_phi_m:
            ax_1.semilogy(
                np.arange(len(self.phi_d)),
                self.phi_m, 'r', lw=2,
                label="$\phi_m$"
            )

        if plot_small_smooth or plot_small:
            ax_1.semilogy(np.arange(
                len(self.phi_d)),
                self.phi_m_small, 'ro',
                label="small"
            )
        if plot_small_smooth or plot_smooth:
            ax_1.semilogy(np.arange(
                len(self.phi_d)),
                self.phi_m_smooth_x, 'rx',
                label="smooth_x"
            )
            ax_1.semilogy(np.arange(
                len(self.phi_d)),
                self.phi_m_smooth_y, 'rx',
                label="smooth_y"
            )
            ax_1.semilogy(np.arange(
                len(self.phi_d)),
                self.phi_m_smooth_z, 'rx',
                label="smooth_z"
            )

        ax.legend(loc=1)
        ax_1.legend(loc=2)

        ax.plot(np.r_[ax.get_xlim()[0], ax.get_xlim()[1]],
                np.ones(2) * self.target_misfit, 'k:')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$\phi_d$")
        ax_1.set_ylabel("$\phi_m$", color='r')
        ax_1.tick_params(axis='y', which='both', colors='red')

        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

    def plot_tikhonov_curves(self, fname=None, dpi=200):

        self.target_misfit = self.invProb.dmisfit.prob.survey.nD / 2.
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

        ax1.plot(self.beta, self.phi_d, 'k-', lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel("$\\beta$", fontsize=14)
        ax1.set_ylabel("$\phi_d$", fontsize=14)

        ax2.plot(self.beta, self.phi_m, 'k-', lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel("$\\beta$", fontsize=14)
        ax2.set_ylabel("$\phi_m$", fontsize=14)

        ax3.plot(self.phi_m, self.phi_d, 'k-', lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel("$\phi_m$", fontsize=14)
        ax3.set_ylabel("$\phi_d$", fontsize=14)

        if self.i_target is not None:
            ax1.plot(self.beta[self.i_target], self.phi_d[self.i_target], 'k*', ms=10)
            ax2.plot(self.beta[self.i_target], self.phi_m[self.i_target], 'k*', ms=10)
            ax3.plot(self.phi_m[self.i_target], self.phi_d[self.i_target], 'k*', ms=10)

        for ax in [ax1, ax2, ax3]:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """
        Saves inversion parameters at every iteraion.
    """

    # Initialize the output dict
    outDict = None
    outDict = {}
    saveOnDisk = False

    def initialize(self):
        print("SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(self.fileName))

    def endIter(self):

        # regCombo = ["phi_ms", "phi_msx"]

        # if self.prob[0].mesh.dim >= 2:
        #     regCombo += ["phi_msy"]

        # if self.prob[0].mesh.dim == 3:
        #     regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = None
        iterDict = {}

        # Save the data.
        iterDict['iter'] = self.opt.iter
        iterDict['beta'] = self.invProb.beta
        iterDict['phi_d'] = self.invProb.phi_d
        iterDict['phi_m'] = self.invProb.phi_m

        # for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
        #     iterDict[label] = fcts(self.invProb.model)

        iterDict['f'] = self.opt.f
        iterDict['m'] = self.invProb.model
        iterDict['dpred'] = self.invProb.dpred

        if hasattr(self.reg.objfcts[0], 'eps_p') is True:
            iterDict['eps_p'] = self.reg.objfcts[0].eps_p
            iterDict['eps_q'] = self.reg.objfcts[0].eps_q

        if hasattr(self.reg.objfcts[0], 'norms') is True:
            iterDict['lps'] = self.reg.objfcts[0].norms[0][0]
            iterDict['lpx'] = self.reg.objfcts[0].norms[0][1]

        # Save the file as a npz
        if self.saveOnDisk:

            np.savez('{:03d}-{:s}'.format(self.opt.iter, self.fileName), iterDict)

        self.outDict[self.opt.iter] = iterDict


class Update_IRLS(InversionDirective):

    f_old = 0
    f_min_change = 1e-2
    beta_tol = 1e-1
    beta_ratio_l2 = None
    prctile = 100
    chifact_start = 1.
    chifact_target = 1.

    # Solving parameter for IRLS (mode:2)
    IRLSiter = 0
    minGNiter = 1
    maxIRLSiter = 20
    iterStart = 0
    sphericalDomain = False

    # Beta schedule
    updateBeta = True
    betaSearch = True
    coolingFactor = 2.
    coolingRate = 1
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

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD

            self._target = nD*0.5*self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    @property
    def start(self):
        if getattr(self, '_start', None) is None:
            if isinstance(self.survey, list):
                self._start = 0
                for survey in self.survey:
                    self._start += survey.nD*0.5*self.chifact_start

            else:

                self._start = self.survey.nD*0.5*self.chifact_start
        return self._start

    @start.setter
    def start(self, val):
        self._start = val

    def initialize(self):

        if self.mode == 1:

            self.norms = []
            for reg in self.reg.objfcts:
                self.norms.append(reg.norms)
                reg.norms = np.c_[2., 2., 2., 2.]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model

        for reg in self.reg.objfcts:
            for comp in reg.objfcts:
                self.f_old += np.sum(comp.f_m**2. / (comp.f_m**2. + comp.epsilon**2.)**(1 - comp.norm/2.))

        self.phi_dm = []
        self.phi_dmx = []
        # Look for cases where the block models in to be scaled
        for prob in self.prob:

            if getattr(prob, 'coordinate_system', None) is not None:
                if prob.coordinate_system == 'spherical':
                    self.sphericalDomain = True

        if self.sphericalDomain:
            self.angleScale()

    def endIter(self):

        if self.sphericalDomain:
            self.angleScale()

        # Check if misfit is within the tolerance, otherwise scale beta
        if np.all([
                np.abs(1. - self.invProb.phi_d / self.target) > self.beta_tol,
                self.updateBeta,
                self.mode != 1
        ]):

            ratio = (self.target / self.invProb.phi_d)

            if ratio > 1:
                ratio = np.mean([2.0, ratio])

            else:
                ratio = np.mean([0.75, ratio])

            self.invProb.beta = self.invProb.beta * ratio

            if np.all([self.mode != 1, self.betaSearch]):
                print("Beta search step")
                # self.updateBeta = False
                # Re-use previous model and continue with new beta
                self.invProb.model = self.reg.objfcts[0].model
                self.opt.xc = self.reg.objfcts[0].model
                self.opt.iter -= 1
                return

        elif np.all([self.mode == 1, self.opt.iter % self.coolingRate == 0]):

            self.invProb.beta = self.invProb.beta / self.coolingFactor

        phim_new = 0
        for reg in self.reg.objfcts:
            for comp in reg.objfcts:
                phim_new += np.sum(
                    comp.f_m**2. /
                    (comp.f_m**2. + comp.epsilon**2.)**(1 - comp.norm/2.)
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
        if np.all([
            (self.opt.iter-self.iterStart) % self.minGNiter == 0,
            self.mode != 1
        ]):

            if self.fix_Jmatrix:
                print(">> Fix Jmatrix")
                self.invProb.dmisfit.prob.fix_Jmatrix = True

            # Check for maximum number of IRLS cycles
            if self.IRLSiter == self.maxIRLSiter:
                if not self.silent:
                    print(
                        "Reach maximum number of IRLS cycles:" +
                        " {0:d}".format(self.maxIRLSiter)
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

            self.IRLSiter += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:

                # If comboObj, go down one more level
                for comp in reg.objfcts:
                    comp.stashedR = None

            for dmis in self.dmisfit.objfcts:
                if getattr(dmis, 'stashedR', None) is not None:
                    dmis.stashedR = None

            # Compute new model objective function value

            phi_m_new = []
            for reg in self.reg.objfcts:
                phi_m_new += [reg(self.invProb.model)]

            self.f_change = np.abs(self.f_old - phim_new) / self.f_old

            if not self.silent:
                print("delta phim: {0:6.3e}".format(self.f_change))

            # Check if the function has changed enough
            if np.all([
                self.f_change < self.f_min_change,
                self.IRLSiter > 1,
                np.abs(1. - self.invProb.phi_d / self.target) < self.beta_tol
            ]):

                print(
                    "Minimum decrease in regularization." +
                    "End of IRLS"
                    )
                self.opt.stopNextIteration = True
                return

            self.f_old = phim_new

            self.updateBeta = True
            self.invProb.phi_m_last = self.reg(self.invProb.model)

    def startIRLS(self):
        if not self.silent:
            print("Reached starting chifact with l2-norm regularization:" +
                  " Start IRLS steps...")

        self.mode = 2

        if getattr(self.opt, 'iter', None) is None:
            self.iterStart = 0
        else:
            self.iterStart = self.opt.iter

        self.invProb.phi_m_last = self.reg(self.invProb.model)

        # Either use the supplied epsilon, or fix base on distribution of
        # model values
        for reg in self.reg.objfcts:

            if getattr(reg, 'eps_p', None) is None:

                reg.eps_p = np.percentile(
                                np.abs(reg.mapping*reg._delta_m(
                                    self.invProb.model)
                                ), self.prctile
                            )

            if getattr(reg, 'eps_q', None) is None:

                reg.eps_q = np.percentile(
                                np.abs(reg.mapping*reg._delta_m(
                                    self.invProb.model)
                                ), self.prctile
                            )

        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms in zip(self.reg.objfcts, self.norms):
            reg.norms = norms

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()

        # Print to screen
        for reg in self.reg.objfcts:
            if not self.silent:
                print("eps_p: " + str(reg.eps_p) +
                      " eps_q: " + str(reg.eps_q))

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
            obj.scales = np.ones(obj.scales.shape)*max_p/var

    def validate(self, directiveList):
        # check if a linear preconditioner is in the list, if not warn else
        # assert that it is listed after the IRLS directive
        dList = directiveList.dList
        self_ind = dList.index(self)
        lin_precond_ind = [
            isinstance(d, UpdatePreconditioner) for d in dList
        ]

        if any(lin_precond_ind):
            assert(lin_precond_ind.index(True) > self_ind), (
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
            regDiag += reg.deriv2(m).diagonal()

        # Deal with the linear case
        if getattr(self.opt, 'JtJdiag', None) is None:

            print("Approximated diag(JtJ) with linear operator")

            JtJdiag = np.zeros_like(self.invProb.model)
            for prob, dmisfit in zip(self.prob, self.dmisfit.objfcts):

                    if getattr(prob, 'getJtJdiag', None) is None:
                        assert getattr(prob, 'getJ', None) is not None, (
                        "Problem does not have a getJ attribute." +
                        "Cannot form the sensitivity explicitely"
                        )
                        JtJdiag += np.sum(np.power((dmisfit.W*prob.getJ(m)), 2), axis=0)
                    else:
                        JtJdiag += prob.getJtJdiag(m, W=dmisfit.W)

            self.opt.JtJdiag = JtJdiag

        diagA = self.opt.JtJdiag + self.invProb.beta*regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.
        PC = Utils.sdiag((diagA))

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
        # Assumes that opt.JtJdiag has been updated or static
        diagA = self.opt.JtJdiag + self.invProb.beta*regDiag
        diagA[diagA != 0] = diagA[diagA != 0] ** -1.
        PC = Utils.sdiag((diagA))
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
                self.k = int(self.survey.nD/10)

            def JtJv(v):

                Jv = self.prob.Jvec(m, v)

                return self.prob.Jtvec(m, Jv)

            JtJdiag = Utils.diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag


class UpdateSensitivityWeights(InversionDirective):
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

        # Send a copy of JtJdiag for the preconditioner
        self.updateOpt()

        # Update the regularization
        self.updateReg()

    def getJtJdiag(self):
        """
            Compute explicitely the main diagonal of JtJ
            Good for any problem where J is formed explicitely
        """
        self.JtJdiag = []
        m = self.invProb.model

        for prob, dmisfit in zip(
            self.prob,
            self.dmisfit.objfcts
        ):

            if getattr(prob, 'getJtJdiag', None) is None:
                assert getattr(prob, 'getJ', None) is not None, (
                    "Problem does not have a getJ attribute." +
                    "Cannot form the sensitivity explicitely"
                )

                self.JtJdiag += [mkvc(np.sum((dmisfit.W*prob.getJ(m))**(2.), axis=0))]
            else:
                self.JtJdiag += [prob.getJtJdiag(m, W=dmisfit.W)]

        return self.JtJdiag

    def getWr(self):
        """
            Take the diagonal of JtJ and return
            a normalized sensitivty weighting vector
        """

        wr = np.zeros_like(self.invProb.model)
        if self.switch:
            for prob_JtJ, prob, dmisfit in zip(self.JtJdiag, self.prob, self.dmisfit.objfcts):

                wr += prob_JtJ + self.threshold

            wr = wr**0.5
            wr /= wr.max()
        else:
            wr += 1.

        return wr

    def updateReg(self):
        """
            Update the cell weights with the approximated sensitivity
        """

        for reg in self.reg.objfcts:
            reg.cell_weights = reg.mapping * (self.wr)

    def updateOpt(self):
        """
            Update a copy of JtJdiag to optimization for preconditioner
        """
        # if self.ComboMisfitFun:
        JtJdiag = np.zeros_like(self.invProb.model)
        for prob, JtJ, dmisfit in zip(
            self.prob, self.JtJdiag, self.dmisfit.objfcts
        ):

            JtJdiag += JtJ

        self.opt.JtJdiag = JtJdiag


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
        nC = int(len(x)/3)

        xyz = Utils.matutils.spherical2cartesian(x.reshape((nC, 3), order='F'))
        m = Utils.matutils.cartesian2spherical(xyz.reshape((nC, 3), order='F'))

        self.invProb.model = m

        for prob in self.prob:
            prob.model = m

        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        nC = int(len(x)/3)

        # Convert to cartesian than back to avoid over rotation
        xyz = Utils.matutils.spherical2cartesian(x.reshape((nC, 3), order='F'))
        m = Utils.matutils.cartesian2spherical(xyz.reshape((nC, 3), order='F'))

        self.invProb.model = m

        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last

        for prob in self.prob:
            prob.model = m

        self.opt.xc = m


class JointInversion_Directive(InversionDirective):
    '''
        Directive for joint inversions. Sets Printers and StoppingCriteria.
        
        Methods assume we are working with two models.
    '''
    class JointInversionPrinters(IterationPrinters):
        betas = {
            "title": "betas", "value": lambda M: ["{:.2e}".format(elem) 
            for elem in M.parent.betas], "width": 26,
            "format":   "%s"
        }
        lambd = {
            "title": "lambda", "value": lambda M: M.parent.lambd, "width": 10,
            "format":   "%1.2e"
        }
        phi_d_joint = {
            "title": "phi_d", "value": lambda M: ["{:.2e}".format(elem) 
            for elem in M.parent.phi_d_joint], "width": 26,
            "format":   "%s"
        }
        phi_m_joint = {
            "title": "phi_m", "value": lambda M: ["{:.2e}".format(elem) 
            for elem in M.parent.phi_m_joint], "width": 26,
            "format":   "%s"
        }
        phi_c = {
            "title": "phi_c", "value": lambda M: M.parent.phi_c, "width": 10,
            "format":   "%1.2e"
        }
        ratio_x = {
            "title": "ratio_x", "value": lambda M: 1 if M.iter==0 else norm(M.xc-M.x_last) / norm(M.x_last),
            "width": 10, "format": "%1.2e"
        }        
        iterationCG = {
            "title": "iterCG", "value": lambda M: M.cg_count, "width": 10, "format": "%3d"
        }
        
    printers = [
            IterationPrinters.iteration, JointInversionPrinters.betas, 
            JointInversionPrinters.lambd, IterationPrinters.f, 
            JointInversionPrinters.phi_d_joint, JointInversionPrinters.phi_m_joint,
            JointInversionPrinters.phi_c, JointInversionPrinters.iterationCG, 
            JointInversionPrinters.ratio_x
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
        assert(self_ind==0), ('The JointInversion_Directive must be first.')
        
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
            
            
class Adaptive_Beta_Reweighting(InversionDirective):
    """
    Adaptively update trade-off parameters in Joint Inversions.
    
    This directive will allow the inversion to run for a few iterations (default is 5),
    and then starts checking if the current data misfits are close to the target.
    
    If data misfit is less than target, we will increase the tradeoff parameter.
    Else, if the data misfit is greater than target, we will decrease the tradeoff parameter.
    
    The rate of increase or decrease can be adjusted (default is 0.01).
    
    We allow for a tolerance range for target misift (default is +- 0.1).
    
    If norm(self.opt.xc - self.opt.x_last) / norm(self.opt.x_last) < self.tol_ratioX,
    it will stop the inversion.
    """
    chifact = 1.
    phi_d_star = []   
    start_iter = 5
    alpha = 0.01 # reweighting coefficient
    dmis_tol = 0.1 # tolerance rate from target data misfit
    tol_ratioX = 1e-5
    
    @property
    def targets(self):
        if getattr(self, '_targets', None) is None:
            if not self.phi_d_star:
                self.phi_d_star = [0.5*survey.nD for survey in self.survey]
        self._targets = [self.chifact*target for target in self.phi_d_star]            
        return self._targets

    @targets.setter
    def targets(self, val):
        assert len(val) == 2, 'val must have two targets.'
        self._targets = val
        
    def initialize(self):
        self._targets = self.targets
        self.betas = self.invProb.betas
        
    def endIter(self):
        ### allow inversion to run for a few iterations before adapting
        ### tradeoff parameters
        if self.opt.iter <= self.start_iter:
            return
        else:
            target_met = []
            for i, phid in enumerate(self.invProb.phi_d_joint):
                if phid > (1+self.dmis_tol)*self._targets[i]:
                    self.betas[i] = (1-self.alpha)*self.betas[i]
                    target_met.append(False)
                elif phid < (1-self.dmis_tol)*self._targets[i]:
                    self.betas[i] = (1+self.alpha)*self.betas[i]
                    target_met.append(False)
                else:
                    target_met.append(True)
                    continue
        self.invProb.betas = self.betas
        self.reg.multipliers[:-1] = self.betas
        
        if all(target_met):
            if norm(self.opt.xc - self.opt.x_last) / norm(self.opt.x_last) < self.tol_ratioX:
                print("stopping criteria met: ", norm(self.opt.xc - self.opt.x_last) 
                                                / norm(self.opt.x_last))
                self.opt.stopNextIteration = True