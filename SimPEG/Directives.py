from __future__ import print_function
from . import Utils
import numpy as np
import scipy.sparse as sp
import warnings
from . import Maps
from .PF import Magnetics, MagneticsDriver
from . import Regularization
from . import Mesh
from . import ObjectiveFunction

class InversionDirective(object):
    """InversionDirective"""

    debug = False    #: Print debugging information

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
        if not isinstance(value, ObjectiveFunction.ComboObjectiveFunction):
            value = 1*value  # turn it into a combo objective function
        self._dmisfit = value

    @property
    def survey(self):
        # if isinstance(self.dmisfit, ObjectiveFunction.ComboObjectiveFunction):
        return [objfcts.survey for objfcts in self.dmisfit.objfcts]

        # else:
        #     return self.dmisfit.survey

    @property
    def prob(self):
        # if isinstance(self.dmisfit, ObjectiveFunction.ComboObjectiveFunction):
        return [objfcts.prob for objfcts in self.dmisfit.objfcts]

        # else:
        #     return self.dmisfit.prob

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass


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
        t = x0.dot(self.dmisfit.deriv2(m, x0, f=f))
        b = x0.dot(self.reg.deriv2(m, v=x0))
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

                # Check if it is a ComboObjective
                if isinstance(self.dmisfit, ObjectiveFunction.ComboObjectiveFunction):
                    self.phi_d_star = 0.5 * self.dmisfit.objfcts[0].survey.nD
                else:
                    self.phi_d_star = 0.5 * self.survey.nD

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

    def initialize(self):

        if getattr(self, 'mapping', None) is None:
            return self.mapPair()
        print("SimPEG.SaveModelEveryIteration will save your models" +
              " in UBC format as: '###-{0!s}.sus'".format(self.fileName))

    def endIter(self):

        if not self.replace:
            fileName = self.fileName + "Iter" + str(self.opt.iter)
        else:
            fileName = self.fileName

        count = -1
        for prob, survey, reg in zip(self.prob, self.survey, self.reg.objfcts):

            count += 1

            if getattr(prob, 'mapping', None) is not None:
                xc = prob.mapping() * self.opt.xc

            else:
                xc = self.opt.xc


            # # Save predicted data
            # if len(self.prob) > 1:
            #     Magnetics.writeUBCobs(fileName + "Prob" + str(count) + '.pre', survey, survey.dpred(m=self.opt.xc))

            # else:
            #     Magnetics.writeUBCobs(fileName + '.pre', survey, survey.dpred(m=self.opt.xc))

            # Save model
            if not isinstance(prob, Magnetics.MagneticVector):

                print()
                # Mesh.TensorMesh.writeModelUBC(reg.mesh,
                #                               fileName + '.sus', self.mapping * xc)
            else:

                if prob.coordinate_system == 'spherical':
                    vec_xyz = Magnetics.atp2xyz(xc)
                else:
                    vec_xyz = xc

                nC = self.mapping.shape[1]

                vec_x = self.mapping * vec_xyz[:nC]
                vec_y = self.mapping * vec_xyz[nC:2*nC]
                vec_z = self.mapping * vec_xyz[2*nC:]

                vec = np.c_[vec_x, vec_y, vec_z]

                m_pst = Magnetics.xyz2pst(vec, self.survey[0].srcField.param)
                m_ind = m_pst.copy()
                m_ind[:, 1:] = 0.
                m_ind = Magnetics.pst2xyz(m_ind, self.survey[0].srcField.param)

                m_rem = m_pst.copy()
                m_rem[:, 0] = 0.
                m_rem = Magnetics.pst2xyz(m_rem, self.survey[0].srcField.param)

                MagneticsDriver.writeVectorUBC(self.prob[0].mesh, fileName + '_VEC.fld', vec)

                if self.saveComp:
                    MagneticsDriver.writeVectorUBC(self.prob[0].mesh, fileName + '_IND.fld', m_ind)
                    MagneticsDriver.writeVectorUBC(self.prob[0].mesh, fileName + '_REM.fld', m_rem)


class SaveOutputEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    def initialize(self):
        print("SimPEG.SaveOutputEveryIteration will save your inversion progress as: '###-{0!s}.txt'".format(self.fileName))
        f = open(self.fileName+'.txt', 'w')
        f.write("  #     beta     phi_d     phi_m       f\n")
        f.close()

    def endIter(self):
        f = open(self.fileName+'.txt', 'a')
        f.write(' {0:3d} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e}\n'.format(self.opt.iter, self.invProb.beta, self.invProb.phi_d, self.invProb.phi_m, self.opt.f))
        f.close()


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
        outDict['phi_ms'] = self.reg._evalSmall(self.invProb.model)
        outDict['phi_mx'] = self.reg._evalSmoothx(self.invProb.model)
        outDict['phi_my'] = self.reg._evalSmoothy(self.invProb.model) if self.prob.mesh.dim >= 2 else 'NaN'
        outDict['phi_mz'] = self.reg._evalSmoothz(self.invProb.model) if self.prob.mesh.dim == 3 else 'NaN'
        outDict['f'] = self.opt.f
        outDict['m'] = self.invProb.model
        outDict['dpred'] = self.invProb.dpred

        # Save the file as a npz
        np.savez('{:03d}-{:s}'.format(self.opt.iter,self.fileName), outDict)


class Update_IRLS(InversionDirective):

    gamma = None
    phi_d_last = None
    f_old = None
    f_min_change = 1e-2
    beta_tol = 5e-2
    prctile = 95
    chifact = 1.

    # Solving parameter for IRLS (mode:2)
    IRLSiter = 0
    minGNiter = 5
    maxIRLSiter = 10
    iterStart = 0

    # Beta schedule
    coolingFactor = 2.
    coolingRate = 1
    ComboRegFun = False
    ComboMisfitFun = False

    updateBeta = True

    mode = 1
    scale_m = False
    phi_m_last = None
    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            if isinstance(self.survey, list):
                self._target = 0
                for survey in self.survey:
                    self._target += survey.nD*0.5*self.chifact

            else:

                self._target = self.survey.nD*0.5*self.chifact
        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):

        if self.mode == 1:

            self.norms = []
            for reg in self.reg.objfcts:
                self.norms.append(reg.norms)
                reg.norms = [2., 2., 2., 2.]
                reg.model = self.invProb.model

        # Update the model used by the regularization
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model

        # Look for cases where the block models in to be scaled
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    self.scale_m = True

                # if np.all([prob.coordinate_system == 'cartesian', len(self.prob) > 1]):
                #     self.scale_m = True

        if self.scale_m:
            self.regScale()

    def endIter(self):

                # Adjust scales for MVI-S
        # if self.ComboMisfitFun:
        if self.scale_m:
            self.regScale()
        # Update the model used by the regularization
        phi_m_last = []
        for reg in self.reg.objfcts:
            reg.model = self.invProb.model
            phi_m_last += [reg(self.invProb.model)]



        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.target, self.mode == 1]):
            print("Convergence with smooth l2-norm regularization: Start IRLS steps...")

            self.mode = 2
            self.coolingFactor = 1.
            self.coolingRate = 1
            self.iterStart = self.opt.iter
            self.phi_d_last = self.invProb.phi_d
            self.invProb.phi_m_last = self.reg(self.invProb.model)

            # Either use the supplied epsilon, or fix base on distribution of
            # model values

            for reg in self.reg.objfcts:

                if getattr(reg, 'eps_p', None) is None:

                    mtemp = reg.mapping * self.invProb.model
                    reg.eps_p = np.percentile(np.abs(mtemp), self.prctile)

                if getattr(reg, 'eps_q', None) is None:
                    mtemp = reg.mapping * self.invProb.model
                    reg.eps_q = np.percentile(np.abs(reg.regmesh.cellDiffxStencil*mtemp), self.prctile)

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

        # Beta Schedule
        if np.all([self.opt.iter > 0, self.opt.iter % self.coolingRate == 0]):
            if self.debug: print('BetaSchedule is cooling Beta. Iteration: {0:d}'.format(self.opt.iter))
            self.invProb.beta /= self.coolingFactor

        # Only update after GN iterations
        if np.all([(self.opt.iter-self.iterStart) % self.minGNiter == 0, self.mode == 2]):


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

            print("Regularization decrease: {0:6.3e}".format((self.f_change)))
            # Check if the function has changed enough
            if self.f_change < self.f_min_change and self.IRLSiter > 1:
                print("Minimum decrease in regularization. End of IRLS")
                self.opt.stopNextIteration = True
                return
            else:
                self.f_old = phim_new

            # Update gamma to scale the regularization between IRLS iterations

            for reg, phim_old, phim_now in zip(self.reg.objfcts, phi_m_last, phi_m_new):

                gamma = phim_old / phim_now

                # If comboObj, go down one more level
                for comp in reg.objfcts:
                    comp.gamma = gamma


            # Check if misfit is within the tolerance, otherwise scale beta
            val = self.invProb.phi_d / self.target

            if np.all([np.abs(1.-val) > self.beta_tol, self.updateBeta]):

                self.invProb.beta = (self.invProb.beta * self.target /
                                     self.invProb.phi_d)

    def regScale(self):
        """
            Update the scales used by regularization for the
            different block of models
        """

        # Currently implemented for MVI-S only
        max_p = []
        for reg in self.reg.objfcts[0].objfcts:
            eps_p = reg.epsilon
            norm_p = 2#self.reg.objfcts[0].norms[0]
            f_m = abs(reg.f_m)
            max_p += [np.max(eps_p**(1-norm_p/2.)*f_m /
                           (f_m**2. + eps_p**2.)**(1-norm_p/2.))]

        max_p = np.asarray(max_p)

        max_s = [np.pi, np.pi]
        for obj, var in zip(self.reg.objfcts[1:], max_s):


            # for reg in obj.objfcts[1:]:
            #     eps_s = reg.epsilon
            #     norm_s = 2#self.reg.objfcts[0].norms[0]
            #     f_m = abs(reg.f_m)
            #     max_s += [np.max(eps_s**(1-norm_s/2.)*f_m /
            #                    (f_m**2. + eps_s**2.)**(1-norm_s/2.))]



            # max_s = np.asarray(max_s)
            obj.scale = max_p.max()/var

        # max_s = np.asarray(max_s)
        # for reg in self.reg.objfcts[1:]:
        #     reg.scale = max_p.max()/max_s.max()
            # print(reg.scale)


class UpdatePreCond(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """
    onlyOnStart = False
    mapping = None
    misfitDiag = None

    def initialize(self):

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)

        for reg in self.reg.objfcts:
            # Check if he has wire
            if getattr(reg.mapping, 'P', None) is None:
                regDiag += (reg.W.T*reg.W).diagonal()
            else:
                # He is a snitch!
                regDiag += reg.mapping.P.T*(reg.W.T*reg.W).diagonal()


        # Deal with the linear case
        if getattr(self.opt, 'JtJdiag', None) is None:

            print("Approximated diag(JtJ) with linear operator")
            wd = self.dmisfit.W.diagonal()
            JtJdiag = np.zeros_like(self.invProb.model)

            for prob in self.prob:
                for ii in range(prob.F.shape[0]):
                    JtJdiag += (wd[ii] * prob.F[ii, :])**2.

            self.opt.JtJdiag = JtJdiag

        diagA = self.opt.JtJdiag + self.invProb.beta*regDiag

        PC = Utils.sdiag((diagA)**-1.)
        self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.onlyOnStart is True:
            return

        # Create the pre-conditioner
        regDiag = np.zeros_like(self.invProb.model)

        for reg in self.reg.objfcts:
            # Check if he has wire
            if getattr(reg.mapping, 'P', None) is None:
                regDiag += (reg.W.T*reg.W).diagonal()
            else:
                # He is a snitch!
                regDiag += reg.mapping.P.T*(reg.W.T*reg.W).diagonal()

        # Assumes that opt.JtJdiag has been updated or static
        diagA = self.opt.JtJdiag + self.invProb.beta*regDiag

        PC = Utils.sdiag((diagA)**-1.)
        self.opt.approxHinv = PC


class UpdateSensWeighting(InversionDirective):
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

        # Update inverse problem
        self.update()

        if self.everyIter:
            # Update the regularization
            self.updateReg()

    def endIter(self):

        # Re-initialize the problem for update
        # if self.ComboMisfitFun:
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    prob._S = None
                    prob.model = self.invProb.model

            if isinstance(prob, Magnetics.MagneticAmplitude):
                prob._dfdm = None
                prob._S = None
                prob.model = self.invProb.model

        # Update inverse problem
        self.update()

        if self.everyIter:
            # Update the regularization
            self.updateReg()

    def update(self):

        # Get sum square of columns of J
        self.getJtJdiag()

        # Compute normalized weights
        self.wr = self.getWr()

        # Send a copy of JtJdiag for the preconditioner
        self.updateOpt()

    def getJtJdiag(self):
        """
            Compute explicitely the main diagonal of JtJ
            Good for any problem where J is formed explicitely
        """
        self.JtJdiag = []
        # if self.ComboMisfitFun:
        Phid = []
        Jmax = []

        for dmisfit in self.dmisfit.objfcts:
            # dmisfit.scale=1.
            dynRange = np.abs(dmisfit.survey.dobs.max() -
                              dmisfit.survey.dobs.min())**2.

            Phid += [dmisfit(self.invProb.model)]
            # dmisfit.scale = 1.
            Jmax += [(np.abs(dmisfit.deriv(self.invProb.model)).max())]

        minPhid = np.asarray(Phid).min()
        # print("Jmax ratio: " +str((Jmax[0]/Jmax[1])**0.5))
        for prob, survey, dmisfit, phid in zip(self.prob,
                                               self.survey,
                                               self.dmisfit.objfcts,
                                               Phid):
            nD = survey.nD
            nC = prob.chiMap.shape[0]
            jtjdiag = np.zeros(nC)
            wd = dmisfit.W.diagonal()

            scale = (phid/minPhid)
            print('Phid: ' + str(dmisfit(self.invProb.model)) + 'Scale: '+ str(scale))
            if isinstance(prob, Magnetics.MagneticVector):

                if prob.coordinate_system == 'spherical':
                    for ii in range(nD):

                        jtjdiag += (wd[ii] * prob.F[ii, :] * prob.S)**2.

                    jtjdiag += 1e-10

                elif prob.coordinate_system == 'cartesian':

                    if getattr(prob, 'JtJdiag', None) is None:
                        prob.JtJdiag = np.sum(prob.F**2., axis=0)

                    jtjdiag = prob.JtJdiag.copy()

            elif isinstance(prob, Magnetics.MagneticAmplitude):

                Bxyz_a = prob.Bxyz_a(prob.chiMap * self.invProb.model)

                if prob.coordinate_system == 'spherical':

                    for ii in range(nD):

                        rows = prob.F[ii::nD, :]
                        jtjdiag += (wd[ii]*(np.dot(Bxyz_a[ii, :],
                                    rows * prob.S)))**2.

                elif getattr(prob, '_Mxyz', None) is not None:
                    for ii in range(nD):

                        jtjdiag += (wd[ii]*(np.dot(Bxyz_a[ii, :],
                                                   prob.F[ii::nD, :]*prob.Mxyz)))**2.

                else:
                    for ii in range(nD):

                        jtjdiag += (wd[ii]*(np.dot(Bxyz_a[ii, :],
                                                   prob.F[ii::nD, :])))**2.

            elif isinstance(prob, Magnetics.MagneticIntegral):

                if getattr(prob, 'JtJdiag', None) is None:
                    prob.JtJdiag = np.sum(prob.F**2., axis=0)

                jtjdiag = prob.JtJdiag

            # Apply scale to the deriv and deriv2
            dmisfit.scale = scale

            # if prob.W is not None:

            #     jtjdiag *= prob.W

            self.JtJdiag += [jtjdiag*scale]

        return self.JtJdiag

    def getWr(self):
        """
            Take the diagonal of JtJ and return
            a normalized sensitivty weighting vector
        """

        wr = np.zeros_like(self.invProb.model)

        # if self.ComboMisfitFun:

        for JtJ, prob in zip(self.JtJdiag, self.prob):

            prob_JtJ = JtJ
            if prob.W is not None:

                prob_JtJ *= prob.W

            prob_JtJ = prob_JtJ**0.5
            prob_JtJ /= prob_JtJ.max()



            if getattr(prob.chiMap, 'index', None) is None:

                wr += prob_JtJ
            else:

                wr[prob.chiMap.index] += prob_JtJ

        # wr = wr**0.5
        # wr /= wr.max()

        # # Apply extra weighting
        # for prob in self.prob:
        #     if prob.W is not None:

        #         if getattr(prob.chiMap, 'index', None) is None:
        #             wr *= prob.W
        #         else:

        #             wr[prob.chiMap.index] *= prob.W

        return wr

    def updateReg(self):
        """
            Update the cell weights with the approximated sensitivity
        """

        for reg in self.reg.objfcts:
            reg.cell_weights = reg.mapping * self.wr


    def updateOpt(self):
        """
            Update a copy of JtJdiag to optimization for preconditioner
        """
        # if self.ComboMisfitFun:
        JtJdiag = np.zeros_like(self.invProb.model)
        for prob, JtJ, dmisfit in zip(self.prob, self.JtJdiag, self.dmisfit.objfcts):

            # Check if he has wire
            if getattr(prob.chiMap, 'index', None) is None:
                JtJdiag += JtJ
            else:
                # He is a snitch!
                JtJdiag[prob.chiMap.index] += JtJ

        self.opt.JtJdiag = JtJdiag

        # else:
        #     self.opt.JtJdiag = self.JtJdiag[0]


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
        xyz = Magnetics.atp2xyz(x)
        m = Magnetics.xyz2atp(xyz)

        self.invProb.model = m

        for prob in self.prob:
            prob.model = m

        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        xyz = Magnetics.atp2xyz(x)
        m = Magnetics.xyz2atp(xyz)

        self.invProb.model = m
        self.invProb.phi_m_last = self.reg(m)

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

    def initialize(self):

        # Get current MVI model and update MAI sensitivity
        # if isinstance(self.prob, list):

        m = self.invProb.model.copy()
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    xyz = Magnetics.atp2xyz(prob.chiMap * m)
                    self.jointMVIS = True
                elif prob.coordinate_system == 'cartesian':
                    xyz = prob.chiMap * m

            if isinstance(prob, Magnetics.MagneticAmplitude):
                self.amp = prob.chiMap * m

        for prob in self.prob:
            if isinstance(prob, Magnetics.MagneticAmplitude):
                if self.jointMVIS:
                    prob.jointMVIS = True

                nC = int(prob.chiMap.shape[0])

                mcol = xyz.reshape((nC, 3), order='F')
                amp = np.sum(mcol**2., axis=1)**0.5
                M = Utils.sdiag(1./amp) * mcol

        else:
            assert("This directive needs to used on a ComboObjective")

    def endIter(self):

        # if self.opt.iter % self.minGNiter == 0:
        # Get current MVI model and update magnetization model for MAI
        m = self.invProb.model.copy()
        for prob in self.prob:

            if isinstance(prob, Magnetics.MagneticVector):
                if prob.coordinate_system == 'spherical':
                    xyz = Magnetics.atp2xyz(prob.chiMap * m)

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
            if isinstance(prob, Magnetics.MagneticAmplitude):

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
                    scale = np.abs(xyz).max()/self.amp.max()
                    # print('Scale: '+str(scale))
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
                self.k = int(self.survey.nD/10)

            def JtJv(v):

                Jv = self.prob.Jvec(m, v)

                return self.prob.Jtvec(m, Jv)

            JtJdiag = Utils.diagEst(JtJv, len(m), k=self.k)
            JtJdiag = JtJdiag / max(JtJdiag)

            self.reg.wght = JtJdiag
