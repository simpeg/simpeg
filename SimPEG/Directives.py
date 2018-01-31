from __future__ import print_function
from . import Utils
from . import Regularization, DataMisfit, ObjectiveFunction
from . import Maps
import numpy as np
import matplotlib.pyplot as plt
import warnings


class InversionDirective(object):
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

    @property
    def name(self):
        if getattr(self, '_name', None) is None:
            self._name = 'InversionModel'
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def fileName(self):
        if getattr(self, '_fileName', None) is None:
            from datetime import datetime
            self._fileName = '{0!s}-{1!s}'.format(
                self.name, datetime.now().strftime('%Y-%m-%d-%H-%M')
            )
        return self._fileName

    @fileName.setter
    def fileName(self, value):
        self._fileName = value


class SaveModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    def initialize(self):
        print("SimPEG.SaveModelEveryIteration will save your models as: '###-{0!s}.npy'".format(self.fileName))

    def endIter(self):
        np.save('{0:03d}-{1!s}'.format(
            self.opt.iter, self.fileName), self.opt.xc
        )


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

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

    def plot_misfit_curves(self, fname=None, plot_small_smooth=False):

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
        ax.semilogy(np.arange(len(self.phi_d)), self.phi_d, 'k-', lw=2)
        ax_1.semilogy(np.arange(len(self.phi_d)), self.phi_m, 'r', lw=2)
        if plot_small_smooth:
            ax_1.semilogy(np.arange(len(self.phi_d)), self.phi_m_small, 'ro')
            ax_1.semilogy(np.arange(len(self.phi_d)), self.phi_m_smooth, 'rx')
            ax_1.legend(
                ("$\phi_m$", "small", "smooth"), bbox_to_anchor=(1.5, 1.)
                )

        ax.plot(np.r_[ax.get_xlim()[0], ax.get_xlim()[1]], np.ones(2)*self.target_misfit, 'k:')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$\phi_d$")
        ax_1.set_ylabel("$\phi_m$", color='r')
        for tl in ax_1.get_yticklabels():
            tl.set_color('r')
        plt.show()

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
            ax.set_xscale("log")
            ax.set_yscale("log")
        plt.tight_layout()
        plt.show()
        if fname is not None:
            fig.savefig(fname, dpi=dpi)


class SaveOutputDictEveryIteration(SaveEveryIteration):
    """
        Saves inversion parameters at every iteraion.
    """

    def initialize(self):
        print("SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(self.fileName))

    def endIter(self):

        # Initialize the output dict
        outDict = {}
        # Save the data.
        outDict['iter'] = self.opt.iter
        outDict['beta'] = self.invProb.beta
        outDict['phi_d'] = self.invProb.phi_d
        outDict['phi_m'] = self.invProb.phi_m

        phi_s, phi_x, phi_y, phi_z = 0, 0, 0, 0
        for reg in self.reg.objfcts:
            phi_s += (
                reg.objfcts[0](self.invProb.model) * reg.alpha_s
            )
            phi_x += (
                reg.objfcts[1](self.invProb.model) * reg.alpha_x
            )
            if reg.regmesh.dim > 1:
                phi_y += (
                    reg.objfcts[2](self.invProb.model) * reg.alpha_y
                )

            if reg.regmesh.dim > 2:
                phi_z += (
                    reg.objfcts[3](self.invProb.model) * reg.alpha_z
                )

        outDict['phi_ms'] = phi_s
        outDict['phi_mx'] = phi_x
        outDict['phi_my'] = phi_y
        outDict['phi_mz'] = phi_z
        outDict['f'] = self.opt.f
        outDict['m'] = self.invProb.model
        outDict['dpred'] = self.invProb.dpred

        # Save the file as a npz
        np.savez('{:03d}-{:s}'.format(self.opt.iter, self.fileName), outDict)


class Update_IRLS(InversionDirective):

    gamma = None
    phi_d_last = None
    f_old = None
    f_min_change = 1e-2
    beta_tol = 5e-2
    prctile = 95
    chifact_start = 1.
    chifact_target = 1.

    # Solving parameter for IRLS (mode:2)
    IRLSiter = 0
    minGNiter = 5
    maxIRLSiter = 10
    iterStart = 0

    # Beta schedule
    updateBeta = True
    coolingFactor = 2.
    coolingRate = 1
    ComboObjFun = False
    mode = 1

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

        # Store norms to be applied after the l2 inversion
        self.norms = []
        for reg in self.reg.objfcts:
            self.norms.append(reg.norms)
            reg.norms = [2., 2., 2., 2.]
            reg.model = self.invProb.model

    def endIter(self):

        # Update the model used by the regularization
        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all(
            [self.invProb.phi_d < self.start, self.mode == 1]
        ):
            print("Reached starting chifact with l2-norm regularization: Start IRLS steps...")

            self.mode = 2
            self.iterStart = self.opt.iter
            self.phi_d_last = self.invProb.phi_d
            self.invProb.phi_m_last = self.reg(self.invProb.model)

            # Either use the supplied epsilon, or fix base on distribution of
            # model values

            for reg in self.reg.objfcts:

                if getattr(reg, 'eps_p', None) is None:

                    mtemp = reg.mapping * self.invProb.model
                    reg.eps_p = np.percentile(np.abs(mtemp), self.prctile)
                    print("eps_p caluclate" + str(reg.eps_p))

                if getattr(reg, 'eps_q', None) is None:
                    mtemp = reg.mapping * self.invProb.model
                    reg.eps_q = np.percentile(
                        np.abs(reg.regmesh.cellDiffxStencil*mtemp),
                        self.prctile
                    )

            # Re-assign the norms supplied by user l2 -> lp
            for reg, norms in zip(self.reg.objfcts, self.norms):
                reg.norms = norms
                print("L[p qx qy qz]-norm : " + str(reg.norms))

            # Save l2-model
            self.invProb.l2model = self.invProb.model.copy()

            # Print to screen
            for reg in self.reg.objfcts:
                print("eps_p: " + str(reg.eps_p) +
                      " eps_q: " + str(reg.eps_q))

        # Only update after GN iterations
        if np.all(
            [(self.opt.iter-self.iterStart) % self.minGNiter == 0, self.mode != 1]
        ):

            # Check for maximum number of IRLS cycles
            if self.IRLSiter == self.maxIRLSiter:
                print("Reach maximum number of IRLS cycles: {0:d}".format(self.maxIRLSiter))
                self.opt.stopNextIteration = True
                return

            # phi_m_last = []
            for reg in self.reg.objfcts:

                # # Reset gamma scale
                # phi_m_last += [reg(self.invProb.model)]

                for comp in reg.objfcts:
                    comp.gamma = 1.

            # Remember the value of the norm from previous R matrices
            self.f_old = self.reg(self.invProb.model)

            self.IRLSiter += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:

                # If comboObj, go down one more level
                for comp in reg.objfcts:
                    comp.stashedR = None

            # Compute new model objective function value
            phim_new = self.reg(self.invProb.model)

            phi_m_new = []
            for reg in self.reg.objfcts:
                phi_m_new += [reg(self.invProb.model)]

            # phim_new = self.reg(self.invProb.model)
            self.f_change = np.abs(self.f_old - phim_new) / self.f_old

            print("Phim relative change: {0:6.3e}".format((self.f_change)))
            # Check if the function has changed enough
            if self.f_change < self.f_min_change and self.IRLSiter > 1:
                print("Minimum decrease in regularization. End of IRLS")
                self.opt.stopNextIteration = True
                return
            else:
                self.f_old = phim_new

            # Update gamma to scale the regularization between IRLS iterations

            for reg, phim_old, phim_now in zip(
                self.reg.objfcts, phi_m_last, phi_m_new
            ):

                gamma = phim_old / phim_now

                # If comboObj, go down one more level
                for comp in reg.objfcts:
                    comp.gamma = gamma

            self.updateBeta = True
        # Beta Schedule
        if np.all([self.invProb.phi_d < self.target,
                   self.mode == 2]):
            print("Target chifact overshooted, adjusting beta ...")
            self.mode = 3

        if np.all([self.opt.iter > 0, self.opt.iter % self.coolingRate == 0,
                   self.mode != 3]):

            if self.debug:
                print('BetaSchedule is cooling Beta. Iteration: {0:d}'.format(self.opt.iter))

            self.invProb.beta /= self.coolingFactor

        # Check if misfit is within the tolerance, otherwise scale beta
        if np.all(
            [
                np.abs(1. - self.invProb.phi_d / self.target) > self.beta_tol,
                self.updateBeta,
                self.mode == 3
            ]
        ):

            self.invProb.beta = (self.invProb.beta * self.target /
                                 self.invProb.phi_d)
            self.updateBeta = False

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
    onlyOnStart = False
    mapping = None
    ComboObjFun = False

    def initialize(self):

        if getattr(self.opt, 'approxHinv', None) is None:

            if getattr(self.opt, 'JtJdiag', None) is None:

                JtJdiag = np.zeros_like(self.invProb.model)
                for prob, dmisfit in zip(self.prob, self.dmisfit.objfcts):

                    assert getattr(prob, 'getJ', None) is not None, (
                        "Problem does not have a getJ attribute." +
                        "Cannot form the sensitivity explicitely"
                    )

                    m = self.invProb.model

                    JtJdiag += np.sum((dmisfit.W*prob.getJ(m))**2., axis=0)

                self.opt.JtJdiag = JtJdiag

            # Update the pre-conditioner
            reg_diag = np.zeros_like(self.invProb.model)
            for reg in self.reg.objfcts:
                reg_diag += self.invProb.beta*(reg.W.T*reg.W).diagonal()

            Hdiag = self.opt.JtJdiag + reg_diag

            PC = Utils.sdiag(Hdiag**-1.)
            self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.onlyOnStart is True:
            return

        if getattr(self.opt, 'approxHinv', None) is not None:

            # Update the pre-conditioner
            reg_diag = np.zeros_like(self.invProb.model)
            for reg in self.reg.objfcts:
                reg_diag += self.invProb.beta*(reg.W.T*reg.W).diagonal()

            Hdiag = self.opt.JtJdiag + reg_diag

            PC = Utils.sdiag(Hdiag**-1.)
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

        for prob, survey, dmisfit in zip(
            self.prob,
            self.survey,
            self.dmisfit.objfcts
        ):

            assert getattr(prob, 'getJ', None) is not None, (
                "Problem does not have a getJ attribute." +
                "Cannot form the sensitivity explicitely"
            )

            m = self.invProb.model

            self.JtJdiag += [np.sum((dmisfit.W*prob.getJ(m))**2., axis=0)]

        return self.JtJdiag

    def getWr(self):
        """
            Take the diagonal of JtJ and return
            a normalized sensitivty weighting vector
        """

        wr = np.zeros_like(self.invProb.model)

        for prob_JtJ, prob in zip(self.JtJdiag, self.prob):

            wr += prob_JtJ + self.threshold

        wr = wr**0.5
        wr /= wr.max()

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
