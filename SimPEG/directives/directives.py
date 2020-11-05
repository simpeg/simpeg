from __future__ import print_function

import properties
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import scipy.sparse as sp
from ..data_misfit import BaseDataMisfit
from ..objective_function import ComboObjectiveFunction
from ..maps import SphericalSystem, ComboMap
from ..regularization import (
    BaseComboRegularization,
    BaseRegularization,
    SimpleSmall,
    Small,
    SparseSmall,
    Simple,
    Tikhonov,
    Sparse,
    SimplePGIsmallness,
    PGIsmallness,
    SimplePGIwithRelationshipsSmallness,
    SimplePGI,
    PGI,
    SmoothDeriv,
    SimpleSmoothDeriv,
    SparseDeriv,
    SimplePGIwithRelationships,
)
from ..utils import (
    mkvc,
    setKwargs,
    sdiag,
    diagEst,
    spherical2cartesian,
    cartesian2spherical,
    Zero,
)
from ..utils.code_utils import deprecate_property


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

        # else:
        #     return self.dmisfit.survey

    @property
    def simulation(self):
        """
           Assuming that dmisfit is always a ComboObjectiveFunction,
           return a list of problems for each dmisfit [prob1, prob2, ...]
        """
        return [objfcts.simulation for objfcts in self.dmisfit.objfcts]

    prob = deprecate_property(
        simulation, "prob", new_name="simulation", removal_version="0.15.0"
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
    """BetaEstimate"""

    beta0 = None  #: The initial Beta (regularization parameter)
    beta0_ratio = 1e2  #: estimateBeta0 is used with this ratio
    ninit = 4  #: number of vector for estimation.
    seed = 518936  #: Random Seed

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
        np.random.seed(self.seed)

        if self.debug:
            print("Calculating the beta0 parameter.")

        m = self.invProb.model
        f = self.invProb.getFields(m, store=True, deleteWarmstart=False)

        ratio = []
        x0 = np.random.rand(*m.shape)
        x0 = x0 / np.linalg.norm(x0)
        for i in range(self.ninit - 1):
            x1 = 0.0
            t = 0.0
            i_count = 0
            for mult, dmis in zip(self.dmisfit.multipliers, self.dmisfit.objfcts):
                # check if f is list
                if len(self.dmisfit.objfcts) > 1:
                    x1 += mult * dmis.deriv2(m, x0, f=f[i_count])
                    i_count += 1
                else:
                    x1 = mult * dmis.deriv2(m, x0, f=f)
            x0 = x1 / np.linalg.norm(x1)
        i_count = 0
        t = 0.0
        for mult, dmis in zip(self.dmisfit.multipliers, self.dmisfit.objfcts):
            # check if f is list
            if len(self.dmisfit.objfcts) > 1:
                t += mult * x0.dot(dmis.deriv2(m, x0, f=f[i_count]))
                i_count += 1
            else:
                t = mult * x0.dot(dmis.deriv2(m, x0, f=f))

        x0 = np.random.rand(*m.shape)
        x0 = x0 / np.linalg.norm(x0)
        for i in range(self.ninit - 1):
            x1 = 0.0
            for mult, reg in zip(self.reg.multipliers, self.reg.objfcts):
                x1 += mult * reg.deriv2(m, v=x0)
            x0 = x1 / np.linalg.norm(x1)

        b = 0.0
        for mult, reg in zip(self.reg.multipliers, self.reg.objfcts):
            b += mult * x0.dot(reg.deriv2(m, v=x0))

        self.ratio = t / b
        self.beta0 = self.beta0_ratio * self.ratio

        self.invProb.beta = self.beta0


class BetaSchedule(InversionDirective):
    """BetaSchedule"""

    coolingFactor = 8.0
    coolingRate = 3

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug:
                print(
                    "BetaSchedule is cooling Beta. Iteration: {0:d}".format(
                        self.opt.iter
                    )
                )
            self.invProb.beta /= self.coolingFactor


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


class SaveEveryIteration(InversionDirective):
    """SaveEveryIteration

    This directive saves an array at each iteration. The default
    directory is the current directoy and the models are saved as
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
            f = open(self.fileName + ".txt", "w")
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
        self.phi_m_small.append(phi_s)
        self.phi_m_smooth_x.append(phi_x)
        self.phi_m_smooth_y.append(phi_y)
        self.phi_m_smooth_z.append(phi_z)
        self.phi.append(self.opt.f)

        if self.save_txt:
            f = open(self.fileName + ".txt", "a")
            f.write(
                " {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e} {5:1.4e} "
                "{6:1.4e}  {7:1.4e}  {8:1.4e}\n".format(
                    self.opt.iter,
                    self.beta[self.opt.iter - 1],
                    self.phi_d[self.opt.iter - 1],
                    self.phi_m[self.opt.iter - 1],
                    self.phi_m_small[self.opt.iter - 1],
                    self.phi_m_smooth_x[self.opt.iter - 1],
                    self.phi_m_smooth_y[self.opt.iter - 1],
                    self.phi_m_smooth_z[self.opt.iter - 1],
                    self.phi[self.opt.iter - 1],
                )
            )
            f.close()

    def load_results(self):
        results = np.loadtxt(self.fileName + str(".txt"), comments="#")
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
    beta_search = properties.Bool("Do a beta serarch", default=False)
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
        removal_version="0.15.0",
    )
    updateBeta = deprecate_property(
        update_beta, "updateBeta", new_name="update_beta", removal_version="0.15.0"
    )
    betaSearch = deprecate_property(
        beta_search, "betaSearch", new_name="beta_search", removal_version="0.15.0"
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


class AlphasSmoothEstimate_ByEig(InversionDirective):
    """AlhaEstimate"""

    alpha0 = 1.0  #: The initial Alha (regularization parameter)
    alpha0_ratio = 1e-2  #: estimateAlha0 is used with this ratio
    ninit = 1
    verbose = False
    debug = False

    def initialize(self):
        """
        """
        if getattr(self.invProb.reg.objfcts[0], "objfcts", None) is not None:
            nbr = np.sum(
                [
                    len(self.invProb.reg.objfcts[i].objfcts)
                    for i in range(len(self.invProb.reg.objfcts))
                ]
            )
            smallness = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            (
                                isinstance(regpart, SimplePGIwithRelationshipsSmallness)
                                or isinstance(regpart, SimplePGIsmallness)
                                or isinstance(regpart, PGIsmallness)
                                or isinstance(regpart, SimpleSmall)
                                or isinstance(regpart, Small)
                                or isinstance(regpart, SparseSmall)
                            ),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            smallness = smallness[smallness[:, 2] == 1][:, :2][0]

            if self.debug:
                print(
                    type(self.invProb.reg.objfcts[smallness[0]].objfcts[smallness[1]])
                )

            smoothness = np.r_[
                [
                    (
                        np.r_[
                            i,
                            j,
                            (
                                (
                                    isinstance(regpart, SmoothDeriv)
                                    or isinstance(regpart, SimpleSmoothDeriv)
                                    or isinstance(regpart, SparseDeriv)
                                )
                                and not (
                                    isinstance(regobjcts, SimplePGI)
                                    or isinstance(regobjcts, PGI)
                                    or isinstance(regobjcts, SimplePGIwithRelationships)
                                    or isinstance(regpart, Tikhonov)
                                    or isinstance(regpart, Simple)
                                    or isinstance(regpart, Sparse)
                                )
                            ),
                        ]
                    )
                    for i, regobjcts in enumerate(self.invProb.reg.objfcts)
                    for j, regpart in enumerate(regobjcts.objfcts)
                ]
            ]
            mode = 1
        else:
            nbr = len(self.invProb.reg.objfcts)
            smoothness = np.r_[
                [
                    (
                        isinstance(regpart, SmoothDeriv)
                        or isinstance(regpart, SimpleSmoothDeriv)
                        or isinstance(regpart, SparseDeriv)
                    )
                    for regpart in self.invProb.reg.objfcts
                ]
            ]
            mode = 2

        if not isinstance(self.alpha0_ratio, np.ndarray):
            self.alpha0_ratio = self.alpha0_ratio * np.ones(nbr)

        if not isinstance(self.alpha0, np.ndarray):
            self.alpha0 = self.alpha0 * np.ones(nbr)

        if self.debug:
            print("Calculating the Alpha0 parameter.")

        m = self.invProb.model

        if mode == 2:
            for i in range(nbr):
                ratio = []
                if smoothness[i]:
                    x0 = np.random.rand(*m.shape)
                    x0 = x0 / np.linalg.norm(x0)
                    x1 = np.random.rand(*m.shape)
                    x1 = x1 / np.linalg.norm(x1)
                    for j in range(self.ninit):
                        x0 = self.invProb.reg.objfcts[0].deriv2(m, v=x0)
                        x1 = self.invProb.reg.objfcts[i].deriv2(m, v=x1)
                        x0 = x0 / np.linalg.norm(x0)
                        x1 = x1 / np.linalg.norm(x1)
                    t = x0.dot(self.invProb.reg.objfcts[0].deriv2(m, v=x0))
                    b = x1.dot(self.invProb.reg.objfcts[i].deriv2(m, v=x1))
                    ratio = t / b

                    self.alpha0[i] *= self.alpha0_ratio[i] * (ratio)
                    mtype = self.invProb.reg.objfcts[i]._multiplier_pair
                    setattr(self.invProb.reg, mtype, self.alpha0[i])

        elif mode == 1:
            for i in range(nbr):
                ratio = []
                if smoothness[i, 2]:
                    idx = smoothness[i, :2]
                    if self.debug:
                        print(type(self.invProb.reg.objfcts[idx[0]].objfcts[idx[1]]))
                    x0 = np.random.rand(*m.shape)
                    x0 = x0 / np.linalg.norm(x0)
                    x1 = np.random.rand(*m.shape)
                    x1 = x1 / np.linalg.norm(x1)
                    for j in range(self.ninit):
                        x0 = (
                            self.invProb.reg.objfcts[smallness[0]]
                            .objfcts[smallness[1]]
                            .deriv2(m, v=x0)
                        )
                        x1 = (
                            self.invProb.reg.objfcts[idx[0]]
                            .objfcts[idx[1]]
                            .deriv2(m, v=x1)
                        )
                        x0 = x0 / np.linalg.norm(x0)
                        x1 = x1 / np.linalg.norm(x1)
                    t = x0.dot(
                        self.invProb.reg.objfcts[smallness[0]]
                        .objfcts[smallness[1]]
                        .deriv2(m, v=x0)
                    )
                    b = x1.dot(
                        self.invProb.reg.objfcts[idx[0]].objfcts[idx[1]].deriv2(m, v=x1)
                    )
                    ratio = np.divide(t, b, out=np.zeros_like(t), where=b != 0)

                    self.alpha0[i] *= self.alpha0_ratio[i] * ratio
                    mtype = (
                        self.invProb.reg.objfcts[idx[0]]
                        .objfcts[idx[1]]
                        ._multiplier_pair
                    )
                    setattr(self.invProb.reg.objfcts[idx[0]], mtype, self.alpha0[i])

        if self.verbose:
            print("Alpha scales: ", self.invProb.reg.multipliers)
            if mode == 1:
                for objf in self.invProb.reg.objfcts:
                    print("Alpha scales: ", objf.multipliers)


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
                                isinstance(regpart, SimplePGIwithRelationshipsSmallness)
                                or isinstance(regpart, SimplePGIsmallness)
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
                                isinstance(regpart, SimplePGIwithRelationshipsSmallness)
                                or isinstance(regpart, SimplePGIsmallness)
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
                    self.phi_d_star = np.r_[[0.5 * self.invProb.dmisfit.survey.nD]]

            self._DMtarget = self.chifact * self.phi_d_star
        return self._DMtarget

    @DMtarget.setter
    def DMtarget(self, val):
        self._DMtarget = val

    @property
    def CLtarget(self):
        if not getattr(self.pgi_smallness, "approx_eval", True):
            # if nonlinear prior, compute targer numerically at each GMM update
            samples, _ = self.pgi_smallness.gmm.sample(len(self.pgi_smallness.gmm.vol))
            self.phi_ms_star = self.pgi_smallness(
                mkvc(samples), externalW=self.WeightsInTarget
            )

            self._CLtarget = self.chiSmall * self.phi_ms_star

        elif getattr(self, "_CLtarget", None) is None:
            # the factor of 0.5 is because we do phid = 0.5*|| dpred - dobs||^2
            if self.phi_ms_star is None:
                # Expected value is number of active cells * number of physical
                # properties
                # if self.WeightsInTarget:
                #    self.phi_ms_star = 0.5
                # else:
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
            message = "All targets reached: {}".format(self.AllStop)
            message += " | geophys. misfits: " + "; ".join(
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


class ScalingEstimate_ByEig(InversionDirective):
    """BetaEstimate"""

    Chi0 = 1  #: The initial Beta (regularization parameter)
    Chi0_ratio = 1  #: estimateBeta0 is used with this ratio
    ninit = 1
    verbose = False

    def initialize(self):
        """
           Assume only 2 data misfits
        """

        if self.debug:
            print("Calculating the scaling parameter.")

        if len(self.dmisfit.objfcts) == 1:
            raise Exception("This Directives only applies ot joint inversion")

        ndm = len(self.dmisfit.objfcts)
        self.Chi0 = self.Chi0 * np.ones(ndm)

        m = self.invProb.model
        f = self.invProb.getFields(m, store=True, deleteWarmstart=False)

        ratio = np.zeros((self.ninit, ndm - 1))
        t = np.zeros(ndm)

        for j in range(ndm):
            x0 = np.random.rand(*m.shape)
            x0 = x0 / np.linalg.norm(x0)
            for i in range(self.ninit - 1):
                x0 = self.dmisfit.objfcts[j].deriv2(m, x0, f=f[0])
                x0 = x0 / np.linalg.norm(x0)
            t[j] = x0.dot(self.dmisfit.objfcts[j].deriv2(m, x0, f=f[0]))

        self.ratio = t[0] / t[1:]
        self.Chi0[1:] = self.Chi0[1:] * self.ratio
        self.dmisfit.multipliers = self.Chi0
        self.dmisfit.multipliers /= np.sum(self.dmisfit.multipliers)

        if self.verbose:
            print("Scale Multipliers: ", self.dmisfit.multipliers)


class JointScalingSchedule(InversionDirective):

    verbose = False
    warmingFactor = 1.0
    mode = 1
    chimax = 1e10
    chimin = 1e-10
    UpdateRate = 1

    def initialize(self):

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
            print("initial data misfit scale: ", self.dmisfit.multipliers)

    def endIter(self):

        self.dmlist = self.inversion.directiveList.dList[self.targetclass].dmlist

        if np.any(self.dmlist < self.DMtarget):
            self.mode = 2
        else:
            self.mode = 1

        if self.opt.iter > 0 and self.opt.iter % self.UpdateRate == 0:

            if self.mode == 2:

                if np.all(np.r_[self.dmisfit.multipliers] > self.chimin) and np.all(
                    np.r_[self.dmisfit.multipliers] < self.chimax
                ):

                    # Assume only 2 data misfit
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

        for reg in self.reg.objfcts:
            reg.cell_weights = reg.mapping * (self.wr)
            if getattr(reg, "objfcts", None) is not None:
                for obj in reg.objfcts:
                    obj.cell_weights = obj.mapping * (self.wr)

    def validate(self, directiveList):
        # check if a beta estimator is in the list after setting the weights
        dList = directiveList.dList
        self_ind = dList.index(self)
        beta_estimator_ind = [isinstance(d, BetaEstimate_ByEig) for d in dList]

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
