from __future__ import print_function
from . import Utils
from . import DataMisfit
import numpy as np
import matplotlib.pyplot as plt
import warnings
from .PF import Magnetics
from . import Regularization
from . import Mesh
from . import ObjectiveFunction


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
        # if getattr(self, '_inversion', None) is not None:
        #     warnings.warn(
        #         'InversionDirective {0!s} has switched to a new inversion.'
        #         .format(self.__class__.__name__)
        #     )
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

        x0 = np.random.rand(m.shape[0])
        t = np.dot(x0, self.dmisfit.deriv2(m, x0, f=f))
        b = np.dot(x0, self.reg.deriv2(m, v=x0))
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


class SaveUBCModelEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    replace = True
    saveComp = True
    mapping = None
    vector = False
    mesh = None

    def initialize(self):

        if getattr(self, 'mapping', None) is None:
            return self.mapPair()
        print("SimPEG.SaveModelEveryIteration will save your models" +
              " in UBC format as: '###-{0!s}.mod'".format(self.fileName))

    def endIter(self):

        if not self.replace:
            fileName = self.fileName + "Iter" + str(self.opt.iter)
        else:
            fileName = self.fileName

        count = -1
        for prob, survey, reg in zip(self.prob, self.survey, self.reg.objfcts):

            count += 1

            # if getattr(prob, 'mapping', None) is not None:
            #     xc = prob.mapping * self.opt.xc

            # else:
            xc = self.mapping * self.opt.xc


            # # Save predicted data
            # if len(self.prob) > 1:
            #     Magnetics.writeUBCobs(fileName + "Prob" + str(count) + '.pre', survey, survey.dpred(m=self.opt.xc))

            # else:
            #     Magnetics.writeUBCobs(fileName + '.pre', survey, survey.dpred(m=self.opt.xc))

            # Save model
            if not self.vector:

                if isinstance(self.mesh, Mesh.TreeMesh):
                        Mesh.TreeMesh.writeUBC(
                            self.mesh,
                            fileName + '.msh',
                            models={fileName + '.mod': xc}
                        )

                else:
                    Mesh.TensorMesh.writeModelUBC(self.mesh,
                                              fileName + '.mod', xc)
            else:

                nC = self.mesh.nC

                if prob.coordinate_system == 'spherical':
                    vec_xyz = Utils.matutils.atp2xyz(xc.reshape((int(len(xc)/3), 3), order='F'))
                    theta = xc[nC:2*nC]
                    phi = xc[2*nC:]
                else:
                    vec_xyz = xc
                    atp = Utils.matutils.xyz2atp(xc.reshape((int(len(xc)/3), 3), order='F'))
                    theta = atp[nC:2*nC]
                    phi = atp[2*nC:]

                vec_x = vec_xyz[:nC]
                vec_y = vec_xyz[nC:2*nC]
                vec_z = vec_xyz[2*nC:]

                vec = np.c_[vec_x, vec_y, vec_z]

                m_pst = Utils.matutils.xyz2pst(
                    vec, self.survey[0].srcField.param
                )
                m_ind = m_pst.copy()
                m_ind[:, 1:] = 0.
                m_ind = Utils.matutils.pst2xyz(
                    m_ind, self.survey[0].srcField.param
                )

                m_rem = m_pst.copy()
                m_rem[:, 0] = 0.
                m_rem = Utils.matutils.pst2xyz(
                    m_rem, self.survey[0].srcField.param
                )

                if self.saveComp:
                    if isinstance(self.mesh, Mesh.TreeMesh):
                        Mesh.TreeMesh.writeUBC(
                            self.mesh,
                            fileName + '.msh',
                            models={
                                fileName + '.dip': (np.rad2deg(theta)),
                                fileName + '.azm': ((450 - np.rad2deg(phi)) % 360),
                                fileName + '_TOT.amp': np.sum(vec**2, axis=1)**0.5,
                                fileName + '_IND.amp': np.sum(m_ind**2, axis=1)**0.5,
                                fileName + '_REM.amp': np.sum(m_rem**2, axis=1)**0.5 }
                        )

                        Utils.io_utils.writeVectorUBC(
                            self.mesh,
                            fileName + '.fld',
                            vec
                        )

                    else:
                        Mesh.TensorMesh.writeModelUBC(
                            self.mesh,
                            fileName + '.dip', (np.rad2deg(theta))
                        )
                        Mesh.TensorMesh.writeModelUBC(
                            self.mesh,
                            fileName + '.azm', (450 - np.rad2deg(phi)) % 360
                        )
                        Mesh.TensorMesh.writeModelUBC(
                            self.mesh,
                            fileName + '_TOT.amp', np.sum(vec**2, axis=1)**0.5
                        )
                        Mesh.TensorMesh.writeModelUBC(
                            self.mesh,
                            fileName + '_IND.amp', np.sum(m_ind**2, axis=1)**0.5
                        )
                        Mesh.TensorMesh.writeModelUBC(
                            self.mesh,
                            fileName + '_REM.amp', np.sum(m_rem**2, axis=1)**0.5
                        )
                        Utils.io_utils.writeVectorUBC(
                            self.mesh,
                            fileName + '_VEC.fld', vec
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

            # f_m = reg.objfcts[0].f_m
            # phi_s += np.sum(f_m**2./(f_m**2. + 1e-8)**(1-reg.objfcts[0].norm/2.))

            # f_m = reg.objfcts[1].f_m
            # phi_x += np.sum(f_m**2./(f_m**2. + 1e-8)**(1-reg.objfcts[1].norm/2.))
            phi_s += (
                reg.objfcts[0](self.invProb.model) * reg.alpha_s
            )
            phi_x += (
                reg.objfcts[1](self.invProb.model) * reg.alpha_x
            )

            if reg.regmesh.dim > 1:
                # f_m = reg.objfcts[2].f_m
                # phi_x += np.sum(f_m**2./(f_m**2. + 1e-8)**(1-reg.objfcts[2].norm/2.))
                phi_y += (
                    reg.objfcts[2](self.invProb.model) * reg.alpha_y
                )

            if reg.regmesh.dim > 2:
                # f_m = reg.objfcts[3].f_m
                # phi_x += np.sum(f_m**2./(f_m**2. + 1e-8)**(1-reg.objfcts[3].norm/2.))

                phi_z += (
                    reg.objfcts[3](self.invProb.model) * reg.alpha_z
                )

            # elif reg.regmesh.dim == 3:
            #     phi_y += (
            #         reg.objfcts[2](self.invProb.model) * reg.alpha_y
            #     )


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

        if self.reg.regmesh.dim == 1:
            self.phi_m_smooth = self.phi_m_smooth_x.copy()
        elif self.reg.regmesh.dim == 2:
            self.phi_m_smooth = self.phi_m_smooth_x + self.phi_m_smooth_y
        elif self.reg.regmesh.dim == 3:
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

        fig = plt.figure(figsize = (5, 8))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

        ax1.plot(self.beta, self.phi_d, 'k-', lw=2, ms=4)
        ax1.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax1.set_xlabel("$\\beta$", fontsize = 14)
        ax1.set_ylabel("$\phi_d$", fontsize = 14)

        ax2.plot(self.beta, self.phi_m, 'k-', lw=2)
        ax2.set_xlim(np.hstack(self.beta).min(), np.hstack(self.beta).max())
        ax2.set_xlabel("$\\beta$", fontsize = 14)
        ax2.set_ylabel("$\phi_m$", fontsize = 14)

        ax3.plot(self.phi_m, self.phi_d, 'k-', lw=2)
        ax3.set_xlim(np.hstack(self.phi_m).min(), np.hstack(self.phi_m).max())
        ax3.set_xlabel("$\phi_m$", fontsize = 14)
        ax3.set_ylabel("$\phi_d$", fontsize = 14)

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

    # Initialize the output dict
    outDict = None
    outDict = {}

    def initialize(self):
        print("SimPEG.SaveOutputDictEveryIteration will save your inversion progress as dictionary: '###-{0!s}.npz'".format(self.fileName))

    def endIter(self):

        regCombo = ["phi_ms", "phi_msx"]

        if self.prob[0].mesh.dim >= 2:
            regCombo += ["phi_msy"]

        if self.prob[0].mesh.dim == 3:
            regCombo += ["phi_msz"]

        # Initialize the output dict
        iterDict = None
        iterDict = {}

        # Save the data.
        iterDict['iter'] = self.opt.iter
        iterDict['beta'] = self.invProb.beta
        iterDict['phi_d'] = self.invProb.phi_d
        iterDict['phi_m'] = self.invProb.phi_m

        for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
            iterDict[label] = fcts(self.invProb.model)

        iterDict['f'] = self.opt.f
        iterDict['m'] = self.invProb.model
        iterDict['dpred'] = self.invProb.dpred

        if hasattr(self.reg.objfcts[0], 'eps_p') is True:
            iterDict['eps_p'] = self.reg.objfcts[0].eps_p
            iterDict['eps_q'] = self.reg.objfcts[0].eps_q

        if hasattr(self.reg.objfcts[0], 'norms') is True:
            for objfct in self.reg.objfcts[0].objfcts:
                objfct.stashedR = None

            iterDict['lps'] = self.reg.objfcts[0].norms[0][0]
            iterDict['lpx'] = self.reg.objfcts[0].norms[0][1]

        iterDict['dphisdm'] = self.reg.objfcts[0].alpha_s * self.reg.objfcts[0].objfcts[0].deriv(self.invProb.model)
        iterDict['dphixdm'] = self.reg.objfcts[0].alpha_x * self.reg.objfcts[0].objfcts[1].deriv(self.invProb.model)

        # Save the file as a npz
        self.outDict[self.opt.iter] = iterDict


class SaveIterationsGeoH5(InversionDirective):
    """
        Saves inversion results to a geoh5 file
    """
    # Initialize the output dict
    h5_object = None
    channels = ['model']
    attribute = "model"
    association = "VERTEX"
    sorting = None
    mapping = None

    def initialize(self):
        if self.attribute == "predicted":
            return
        prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.attribute == "mvi_model":
            prop = np.linalg.norm(prop.reshape((-1, 3), order='F'), axis=1)

        elif self.attribute == "mvis_model":
            prop = prop.reshape((-1, 3), order='F')[:, 0]

        for ii, channel in enumerate(self.channels):

            attr = prop[ii::len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            self.h5_object.add_data({
                    f"Initial": {
                        "association":self.association, "values": attr
                    }
                }
            )

        self.h5_object.workspace.finalize()

    def endIter(self):

        if self.attribute == "predicted":
            if getattr(self.dmisfit, 'objfcts', None) is not None:
                dpred = []
                for local_misfit in self.dmisfit.objfcts:
                    dpred.append(np.asarray(
                        local_misfit.survey.dpred(self.invProb.model)
                    ))
                prop = np.hstack(dpred)
            else:
                prop = self.dmisfit.survey.dpred(self.invProb.model)
        else:
            prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.attribute == "mvi_model":
            prop = np.linalg.norm(prop.reshape((-1, 3), order='F'), axis=1)

        elif self.attribute == "mvis_model":
            prop = prop.reshape((-1, 3), order='F')[:, 0]


        for ii, channel in enumerate(self.channels):

            attr = prop[ii::len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            self.h5_object.add_data({
                    f"Iteration_{self.opt.iter}_" + channel: {
                        "association":self.association, "values": attr
                    }
                }
            )
        self.h5_object.workspace.finalize()


class VectorInversion(InversionDirective):
    """
    Control a vector inversion from Cartesian to spherical coordinates
    """

    chifact_target = 2.
    mref = None
    mode = 'cartesian'
    norms = []
    alphas = []

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            nD = 0
            for survey in self.survey:
                nD += survey.nD

            self._target = nD * 0.5 * self.chifact_target

        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):

        if self.mode == 'cartesian':
            self.norms = []
            self.alphas = []
            for reg in self.reg.objfcts:
                self.norms.append(reg.norms)
                reg.norms = np.c_[2., 2., 2., 2.]
                reg.model = self.invProb.model

            self.mref = reg.mref

        for prob in self.prob:
            if getattr(prob, 'coordinate_system', None) is not None:
                prob.coordinate_system = self.mode

    def endIter(self):
        if (self.invProb.phi_d < self.target) and self.mode == 'cartesian':

            self.mode = 'spherical'
            print("Switching MVI to spherical coordinates")

            mstart = Utils.matutils.xyz2atp(self.invProb.model.reshape((-1, 3), order='F'))
            mref = Utils.matutils.xyz2atp(self.mref.reshape((-1, 3), order='F'))

            self.invProb.model = mstart
            self.opt.xc = mstart

            print(self.invProb.model, self.opt.xc)
            nC = mstart.reshape((-1, 3)).shape[0]
            self.opt.lower = np.kron(np.asarray([0, -np.inf, -np.inf]), np.ones(nC))
            self.opt.upper[nC:] = np.inf

            self.reg.mref = mref
            self.reg.model = mstart

            for prob in self.prob:
                if getattr(prob, 'coordinate_system', None) is not None:
                    prob.coordinate_system = self.mode
                    prob.model = mstart

            for ind, (reg_fun, norms) in enumerate(zip(
                    self.reg.objfcts, self.norms
            )):
                reg_fun.norms = norms
                reg_fun.mref = mref
                reg_fun.model = mstart

                if ind > 0:
                    reg_fun.alpha_s = 0
                    reg_fun.space = 'spherical'
                    reg_fun.eps_q = np.pi
            # Create a regularization
            # print(self.reg.objfcts[2].indActive)
            # reg_a = Regularization.Sparse(
            #     self.reg.objfcts[0].mesh,
            #     indActive=self.reg.objfcts[0].indActive,
            #     mapping=self.reg.objfcts[0].mapping,
            #     gradientType=self.reg.objfcts[0].gradientType,
            #     alpha_s=self.reg.objfcts[0].alpha_s,
            #     alpha_x=self.reg.objfcts[0].alpha_x,
            #     alpha_y=self.reg.objfcts[0].alpha_y,
            #     alpha_z=self.reg.objfcts[0].alpha_z,
            # )
            # reg_a.norms = self.norms[0]
            # reg_a.mref = mref
            #
            # reg_t = Regularization.Sparse(
            #     self.reg.objfcts[1].mesh,
            #     indActive=self.reg.objfcts[1].indActive,
            #     mapping=self.reg.objfcts[1].mapping,
            #     gradientType=self.reg.objfcts[1].gradientType,
            #     alpha_s=0,
            #     alpha_x=self.reg.objfcts[1].alpha_x,
            #     alpha_y=self.reg.objfcts[1].alpha_y,
            #     alpha_z=self.reg.objfcts[1].alpha_z,
            # )
            # reg_t.space = 'spherical'
            # reg_t.norms = self.norms[1]
            # reg_t.mref = mref
            # reg_t.eps_q = np.pi
            #
            # reg_p = Regularization.Sparse(
            #     self.reg.objfcts[2].mesh,
            #     indActive=self.reg.objfcts[2].indActive,
            #     mapping=self.reg.objfcts[2].mapping,
            #     gradientType=self.reg.objfcts[2].gradientType,
            #     alpha_s=0,
            #     alpha_x=self.reg.objfcts[2].alpha_x,
            #     alpha_y=self.reg.objfcts[2].alpha_y,
            #     alpha_z=self.reg.objfcts[2].alpha_z,
            # )
            #
            # reg_p.space = 'spherical'
            # reg_p.norms = self.norms[2]
            # reg_p.mref = mref
            # reg_p.eps_q = np.pi
            #
            # # Assemble the three regularization
            # self.reg = reg_a + reg_t + reg_p
            # self.invProb.reg = self.reg

            # Add directives
            directiveList = []
            for directive in self.inversion.directiveList.dList:
                # directive._inversion = None
                if isinstance(directive, SaveIterationsGeoH5):
                    channels = []
                    for channel in directive.channels:
                        channels.append(channel + "_s")
                    directive.channels = channels

                    if directive.attribute == "mvi_model":
                        directive.attribute = "mvis_model"

                    directiveList.append(directive)

                elif isinstance(directive, SaveUBCModelEveryIteration):
                    directiveList.append(directive)

                elif isinstance(directive, Update_IRLS):
                    directive.sphericalDomain = True
                    directive.model = mstart
                    IRLS = directive

                elif isinstance(directive, UpdatePreconditioner):
                    update_Jacobi = directive

            directiveList = [
                ProjSpherical(), IRLS, UpdateSensitivityWeights(), update_Jacobi,
            ] + directiveList

            # update_SensWeight = UpdateSensitivityWeights()
            # update_Jacobi = UpdatePreconditioner()

            # inv = Inversion.BaseInversion(invProb,
            #                               directiveList=[
            #                                   ProjSpherical, IRLS, update_SensWeight,
            #                                   update_Jacobi, save_model, inversion_output
            #                               ])


            self.inversion.directiveList = directiveList
            directiveList[1].endIter()
            directiveList[2].endIter()
            directiveList[3].endIter()


class Update_IRLS(InversionDirective):

    updateGamma = False
    f_old = np.inf
    f_min_change = 1e-2
    beta_tol = 1e-1
    beta_ratio_l2 = None
    prctile = 90
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
    floorEps_p = [1e-8, 1e-8, 1e-8]
    floorEps_q = [1e-8, 1e-8, 1e-8]
    floorEpsEnforced = True
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

    def initialize(self):

        if self.mode == 1:

            self.norms = []
            self.alpha = []
            for reg in self.reg.objfcts:
                self.norms.append(reg.norms)
                self.alpha.append(reg.alpha_s)
                reg.norms = np.c_[2., 2., 2., 2.]
                reg.model = self.invProb.model

                # # Check if using non-simple difference
                # dx = sp.find(reg.regmesh.cellDiffxStencil)[2].max()
                # if dx != 1:
                #     print(dx)
                #     reg.alpha_s = dx**2. #/np.min(reg.regmesh.mesh.hx)**2.

        # Update the model used by the regularization
        # for reg in self.reg.objfcts:
        #     reg.model = self.invProb.model

        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            for comp in reg.objfcts:
                self.f_old += comp(reg.model)

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
                ratio = np.min([2.0, ratio])

            else:
                ratio = np.max([0.75, ratio])

            self.invProb.beta = self.invProb.beta * ratio

            if np.all([self.mode != 1, self.betaSearch]):
                # Re-use previous model and continue with new beta
                self.invProb.model = self.reg.objfcts[0].model
                self.opt.xc = self.reg.objfcts[0].model
                self.opt.iter -= 1
                return

        elif np.all([self.mode == 1, self.opt.iter % self.coolingRate == 0]):

            self.invProb.beta = self.invProb.beta / self.coolingFactor

        phim_new = 0
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            for comp in reg.objfcts:
                phim_new += comp(reg.model)

        # Update the model used by the regularization
        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.target, self.mode == 1]):
            self.startIRLS()

        # Only update after GN iterations
        if np.all([
            (self.opt.iter-self.iterStart) % self.minGNiter == 0,
            self.mode != 1
        ]):
            if self.fix_Jmatrix:
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
            for ii, reg in enumerate(self.reg.objfcts):

                if reg.eps_p > self.floorEps_p[ii] and self.coolEps_p:
                    reg.eps_p /= self.coolEpsFact

                elif self.floorEpsEnforced:
                    reg.eps_p = self.floorEps_p[ii]

                if reg.eps_q > self.floorEps_q[ii] and self.coolEps_q:
                    reg.eps_q /= self.coolEpsFact

                elif self.floorEpsEnforced:
                    reg.eps_q = self.floorEps_q[ii]

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
                reg.eps_q = reg.eps_p


        # Re-assign the norms supplied by user l2 -> lp
        for reg, norms, alpha in zip(self.reg.objfcts, self.norms, self.alpha):
            reg.norms = norms

            reg.alpha_s = alpha

        # Save l2-model
        self.invProb.l2model = self.invProb.model.copy()


    def angleScale(self):
        """
            Update the scales used by regularization for the
            different block of models
        """
        # Currently implemented for MVI-S only
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            eps_p = reg.epsilon
            norm_p = 2  # self.reg.objfcts[0].norms[0]
            f_m = abs(reg.f_m)
            max_p += [np.max(f_m)]

        max_p = np.asarray(max_p).max()

        scales = [2*max_p/np.pi, 2*max_p/np.pi]
        for obj, scale in zip(self.reg.objfcts[1:3], scales):
            obj.scales = np.ones(obj.scales.shape)*scale
        # self.reg.objfcts[0].scales = np.ones(self.reg.objfcts[0].scales.shape)
        # Probably doing rotated obj fun
        if len(self.reg) > 3:

            for obj, scale in zip(self.reg.objfcts[4:], scales):
                obj.scales = np.ones(obj.scales.shape)*scale

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
    misfitDiag = None
    epsilon = 1e-8

    def initialize(self):

        m = self.invProb.model
        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)

        for reg in self.reg.objfcts:
            # # Check if regularization has a projection
            # if getattr(reg.mapping, 'P', None) is None:
            #     regDiag += (reg.W.T*reg.W).diagonal()
            # else:
            #     P = reg.mapping.P
            #     regDiag += (P.T * (reg.W.T * (reg.W * P))).diagonal()
            regDiag += reg.deriv2(m).diagonal()

        # Deal with the linear case
        if getattr(self.opt, 'JtJdiag', None) is None:
            JtJdiag = np.zeros_like(self.invProb.model)
            m = self.invProb.model
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
        PC = Utils.sdiag(diagA)

        self.opt.approxHinv = PC

    def endIter(self):

        # Cool the threshold parameter
        if self.onlyOnStart is True:
            return

        m = self.invProb.model
        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)

        for reg in self.reg.objfcts:
            regDiag += reg.deriv2(m).diagonal()

        # Assumes that opt.JtJdiag has been updated or static
        diagA = self.opt.JtJdiag + self.invProb.beta*regDiag

        diagA[diagA != 0] = diagA[diagA != 0] ** -1.
        PC = Utils.sdiag(diagA)
        self.opt.approxHinv = PC


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

        for prob, dmisfit in zip(
            self.prob,
            self.dmisfit.objfcts
        ):
            m = self.invProb.model

            if getattr(prob, 'getJtJdiag', None) is None:
                assert getattr(prob, 'getJ', None) is not None, (
                    "Problem does not have a getJ attribute." +
                    "Cannot form the sensitivity explicitely"
                )

                self.JtJdiag += [Utils.mkvc(np.sum((dmisfit.W*prob.getJ(m))**(2.), axis=0))]
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


class ProjSpherical(InversionDirective):
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

        xyz = Utils.matutils.atp2xyz(x.reshape((nC, 3), order='F'))
        m = Utils.matutils.xyz2atp(xyz.reshape((nC, 3), order='F'))

        self.invProb.model = m

        for prob in self.prob:
            prob.model = m

        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        nC = int(len(x)/3)

        # Convert to cartesian than back to avoid over rotation
        xyz = Utils.matutils.atp2xyz(x.reshape((nC, 3), order='F'))
        m = Utils.matutils.xyz2atp(xyz.reshape((nC, 3), order='F'))

        self.invProb.model = m

        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]

        self.invProb.phi_m_last = phi_m_last

        for prob in self.prob:
            prob.model = m

        self.opt.xc = m


class JointAmpMVI(InversionDirective):
    """
        Directive controlling the joint inversion of
        magnetic amplitude data and MVI. Use the vector
        magnetization model (M) to update the linear amplitude
        operator.

    """

    amp = None
    minGNiter = 1
    jointMVIS = False
    updateM = False

    def initialize(self):

        # Get current MVI model and update MAI sensitivity
        # if isinstance(self.prob, list):

        m = self.invProb.model.copy()
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    xyz = Magnetics.atp2xyz((prob.chiMap * m).reshape((int(len(m)/3), 3), order='F'))
                    self.jointMVIS = True
                elif prob.coordinate_system == 'cartesian':
                    xyz = prob.chiMap * m

            if isinstance(prob, Magnetics.MagneticAmplitude):
                self.amp = prob.chiMap * m

        for prob in self.prob:
            if isinstance(prob, Magnetics.MagneticAmplitude):
                if self.jointMVIS:
                    prob.jointMVIS = True

                nC = prob.mesh.nC

                mcol = xyz.reshape((nC, 3), order='F')
                amp = np.sum(mcol**2., axis=1)**0.5
                M = Utils.sdiag(1./amp) * mcol

        else:
            assert("This directive needs to used on a ComboObjective")

    def endIter(self):

        # Get current MVI model and update magnetization model for MAI
        m = self.invProb.model.copy()
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    xyz = Magnetics.atp2xyz((prob.chiMap * m).reshape((int(len(m)/3), 3), order='F'))

                elif prob.coordinate_system == 'cartesian':
                    xyz = prob.chiMap * m

            if isinstance(prob, Magnetics.MagneticAmplitude):
                if prob.chiMap.shape[0] == 3*prob.mesh.nC:

                    nC = prob.mesh.nC

                    mcol = (prob.chiMap * m).reshape((nC, 3), order='F')
                    self.amp = np.sum(mcol**2., axis=1)**0.5
                else:

                    self.amp = prob.chiMap * m

        for prob in self.prob:
            if np.all([isinstance(prob, Magnetics.MagneticAmplitude), self.updateM]):

                nC = prob.mesh.nC

                mcol = xyz.reshape((nC, 3), order='F')
                amp = np.sum(mcol**2., axis=1)**0.5

                M = Utils.sdiag(1./amp) * mcol

                prob.M = M
                prob._Mxyz = None
                prob.Mxyz
                # if prob.model is None:
                #     prob.model = prob.chiMap * m

                # ampW = (amp/amp.max() + 1e-2)**-1.
                # prob.W = ampW

            if isinstance(prob, Magnetics.MagneticVector):

                if (prob.coordinate_system == 'cartesian') and (self.amp is not None):

                    ampW = (self.amp/self.amp.max()+1e-2)**-1.
                    ampW = np.r_[ampW, ampW, ampW]

                    # Scale max values
                    prob.W = ampW


class UpdateApproxJtJ(InversionDirective):
    """
        Create approx-sensitivity base weighting using the probing method
    """
    k = None  # Number of probing cycles
    itr = None  # Iteration number to update Wj, or always update if None

    def endIter(self):

        if self.itr is None or self.itr == self.opt.iter:

            m = self.invProb.model
            if self.k is None:

                nD = 0
                for survey in self.survey:
                    nD += survey.nD

                self.k = int(nD/10)

            def JtJv(v):

                Jv = self.prob.Jvec(m, v)

                return self.prob.Jtvec(m, Jv)

            JtJdiag = Utils.diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag


class ScaleComboReg(InversionDirective):
    """
    Directive to take care of re-weighting
    the non-linear magnetic problems.

    """
    # coordinate_system = 'Amp'
    # test = False
    mapping = None
    ComboRegFun = False
    ComboMisfitFun = False
    JtJdiag = None
    everyIter = True

    def initialize(self):

        # for reg in self.reg.objfcts:
        m = self.invProb.model

        scale = np.abs(self.reg.objfcts[0](m)).max()/np.abs(self.reg.objfcts[1](m)).max()
        self.reg.objfcts[1].scale = scale

    def endIter(self):

        m = self.invProb.model

        scale = np.abs(self.reg.objfcts[0].deriv(m)).max()/np.abs(self.reg.objfcts[1].deriv(m)).max()
        self.reg.objfcts[1].scale = scale
