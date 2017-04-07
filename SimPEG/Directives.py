from __future__ import print_function
from . import Utils
import numpy as np
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
        return self.invProb.reg

    @property
    def dmisfit(self):
        return self.invProb.dmisfit

    @property
    def survey(self):
        return self.dmisfit.survey

    @property
    def prob(self):
        return self.dmisfit.prob

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

    mapping = None
    replace = True

    def initialize(self):

        if getattr(self, 'mapping', None) is None:
            return self.mapPair()
        print("SimPEG.SaveModelEveryIteration will save your models" +
              " in UBC format as: '###-{0!s}.sus'".format(self.fileName))

    def endIter(self):

        # Overwrite the file or add interation number
        if not self.replace:
            fileName = self.fileName + str(self.opt.iter)
        else:
            fileName = self.fileName

        Mesh.TensorMesh.writeModelUBC(self.reg.mesh,
                                      fileName + '.sus',
                                      self.mapping*self.opt.xc)

        Magnetics.writeUBCobs(fileName + '.pre', self.survey, self.invProb.dpred)


class SaveUBCVectorsEveryIteration(SaveEveryIteration):
    """SaveModelEveryIteration"""

    mapping = None
    replace = True
    saveComp = False
    spherical = False

    def initialize(self):
        print("SimPEG.SaveModelEveryIteration will save your models" +
              " in UBC format as: '###-{0!s}.sus'".format(self.fileName))

    def endIter(self):

        nC = self.mapping.shape[1]

        if self.spherical:
            vec_pst = Magnetics.atp2xyz(self.opt.xc)
        else:
            vec_pst = self.opt.xc

        vec_p = self.mapping*vec_pst[:nC]
        vec_s = self.mapping*vec_pst[nC:2*nC]
        vec_t = self.mapping*vec_pst[2*nC:]

        vec = np.c_[vec_p, vec_s, vec_t]

        # Overwrite the file or add interation number
        if not self.replace:
            fileName = self.fileName + str(self.opt.iter)
        else:
            fileName = self.fileName

        MagneticsDriver.writeVectorUBC(self.prob.mesh, fileName + '.fld', vec)

        if self.saveComp:
            Mesh.TensorMesh.writeModelUBC(self.prob.mesh,
                                          fileName + '_amp.sus',
                                          self.mapping*self.opt.xc[:nC])
            Mesh.TensorMesh.writeModelUBC(self.prob.mesh,
                                          fileName + '_phi.sus',
                                          self.mapping*self.opt.xc[nC:2*nC])
            Mesh.TensorMesh.writeModelUBC(self.prob.mesh,
                                          fileName + '_theta.sus',
                                          self.mapping*self.opt.xc[2*nC:])


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
    ComboObjFun = False

    updateBeta = True

    mode = 1

    @property
    def target(self):
        if getattr(self, '_target', None) is None:
            self._target = self.survey.nD*0.5*self.chifact
        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def initialize(self):

        # Check if it is a ComboObjective
        if not isinstance(self.reg, Regularization.BaseComboRegularization):

            # It is a Combo objective, so will have to loop
            self.ComboObjFun = True

        if self.mode == 1:

            if self.ComboObjFun:

                self.norms = []
                for reg in self.reg.objfcts:
                    self.norms.append(reg.norms)
                    reg.norms = [2., 2., 2., 2.]

            else:
                # Store assigned norms for later use - start with l2
                self.norms = self.reg.norms
                self.reg.norms = [2., 2., 2., 2.]

    def endIter(self):

        # After reaching target misfit with l2-norm, switch to IRLS (mode:2)
        if np.all([self.invProb.phi_d < self.target, self.mode == 1]):
            print("Convergence with smooth l2-norm regularization: Start IRLS steps...")

            self.mode = 2
            self.coolingFactor = 1.
            self.coolingRate = 1
            self.iterStart = self.opt.iter
            self.phi_d_last = self.invProb.phi_d
            self.invProb.phi_m_last = self.reg(self.invProb.model)

            if getattr(self, 'f_old', None) is None:
                self.f_old = self.reg(self.invProb.model)

            self.coolingFactor = 1.
            self.coolingRate = 1
            self.iterStart = self.opt.iter
            self.phi_d_last = self.invProb.phi_d
            self.invProb.phi_m_last = self.reg(self.invProb.model)

            # print(' iter', self.opt.iter, 'beta',self.invProb.beta,'phid', self.invProb.phi_d)
            if getattr(self, 'f_old', None) is None:
                self.f_old = self.reg(self.invProb.model)

            # Either use the supplied epsilon, or fix base on distribution of
            # model values
            if self.ComboObjFun:

                for reg in self.reg.objfcts:

                    if getattr(reg, 'eps_p', None) is None:

                        mtemp = reg.mapping * self.invProb.model
                        reg.eps_p = np.percentile(np.abs(mtemp), self.prctile)

                    if getattr(reg, 'eps_q', None) is None:
                        mtemp = reg.mapping * self.invProb.model
                        reg.eps_q = np.percentile(np.abs(reg.regmesh.cellDiffxStencil*mtemp), self.prctile)

            else:
                if getattr(self.reg, 'eps_p', None) is None:

                    mtemp = self.reg.mapping * self.invProb.model
                    self.reg.eps_p = np.percentile(np.abs(mtemp), self.prctile)

                if getattr(self.reg, 'eps_q', None) is None:

                    mtemp = self.reg.mapping * self.invProb.model
                    self.reg.eps_q = np.percentile(np.abs(self.reg.regmesh.cellDiffxStencil*mtemp), self.prctile)


            # Re-assign the norms
            if self.ComboObjFun:
                for reg, norms in zip(self.reg.objfcts, self.norms):
                    reg.norms = norms
                    print("L[p qx qy qz]-norm : " + str(reg.norms))

            else:
                self.reg.norms = self.norms
                print("L[p qx qy qz]-norm : " + str(self.reg.norms))


            if self.ComboObjFun:
                    for reg in self.reg.objfcts:
                        reg.model = self.invProb.model

            else:
                self.reg.model = self.invProb.model

            self.reg.l2model = self.invProb.model.copy()

            # Re-assign the norms
            if self.ComboObjFun:
                for reg in self.reg.objfcts:
                    print("eps_p: " + str(reg.eps_p) +
                          " eps_q: " + str(reg.eps_q))

            else:
                print("eps_p: " + str(self.reg.eps_p) + " eps_q: " + str(self.reg.eps_q))


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

            else:
                # Update the model used in the regularization
                if self.ComboObjFun:
                    for reg in self.reg.objfcts:
                        reg.model = self.invProb.model

                else:
                    self.reg.model = self.invProb.model

                self.IRLSiter += 1

            # Reset the regularization matrices so that it is
            # recalculated for current model. Do it to all levels of comboObj
            for reg in self.reg.objfcts:

                # If comboObj, go down one more level
                if self.ComboObjFun:
                    for comp in reg.objfcts:
                        comp.stashedR = None
                        comp.gamma = 1.
                else:
                    reg.stashedR = None
                    reg.gamma = 1.

            # Compute new model objective function value
            phim_new = self.reg(self.invProb.model)

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
            gamma = self.invProb.phi_m_last / phim_new
            for reg in self.reg.objfcts:

                # If comboObj, go down one more level
                if self.ComboObjFun:
                    for comp in reg.objfcts:
                        comp.stashedR = None
                        comp.gamma = gamma
                else:
                    reg.stashedR = None
                    reg.gamma = gamma

            # Check if misfit is within the tolerance, otherwise scale beta
            val = self.invProb.phi_d / self.target

            if np.all([np.abs(1.-val) > self.beta_tol, self.updateBeta]):

                self.invProb.beta = (self.invProb.beta * self.target /
                                     self.invProb.phi_d)

    def validate(self, directiveList):
        # check if a linear preconditioner is in the list, if not warn else
        # assert that it is listed after the IRLS directive
        dList = directiveList.dList
        self_ind = dList.index(self)
        lin_precond_ind = [
            isinstance(d, Update_lin_PreCond) for d in dList
        ]

        if any(lin_precond_ind):
            assert(lin_precond_ind.index(True) > self_ind), (
                "The directive 'Update_lin_PreCond' must be after Update_IRLS "
                "in the directiveList"
            )
        else:
            warnings.warn(
                "Without a Linear preconditioner, convergence may be slow."
            )
        return True


class Update_lin_PreCond(InversionDirective):
    """
    Create a Jacobi preconditioner for the linear problem
    """
    onlyOnStart = False
    mapping = None
    ComboRegFun = False
    ComboMisfitFun = False
    misfitDiag = None

    def initialize(self):

        # Check if it is a ComboObjective
        if not isinstance(self.reg, Regularization.BaseComboRegularization):

            # It is a Combo objective, so will have to loop
            self.ComboRegFun = True

        # Check if it is a ComboObjective
        if isinstance(self.dmisfit, ObjectiveFunction.ComboObjectiveFunction):

            # It is a Combo objective, so will have to loop
            self.ComboMisfitFun = True

        # if getattr(self, 'mapping', None) is None:
        #     self.mapping = Maps.IdentityMap(nP=self.reg.mapping.nP)

        if getattr(self.opt, 'approxHinv', None) is None:

            # Update the pre-conditioner
            if self.ComboRegFun:

                regDiag = []
                for reg in self.reg.objfcts:
                    regDiag.append((reg.W.T*reg.W).diagonal())

                regDiag = np.hstack(regDiag)

            else:

                regDiag = (self.reg.W.T*self.reg.W).diagonal()

            if self.ComboMisfitFun:

                if getattr(self, 'misfitDiag', None) is None:
                    misfitDiag = np.zeros(self.prob.F.shape[1])
                    for misfit in self.dmisfit.objfcts:
                        wd = misfit.W.diagonal()
                        misfitDiag = np.zeros(misfit.prob.F.shape[1])
                        for ii in range(misfit.prob.F.shape[0]):
                            misfitDiag += (wd[ii] * misfit.prob.F[ii, :])**2.

                self.misfitDiag = np.hstack(misfitDiag)

            else:

                if getattr(self, 'misfitDiag', None) is None:
                    wd = self.dmisfit.W.diagonal()
                    self.misfitDiag = np.zeros(self.prob.F.shape[1])
                    for ii in range(self.prob.F.shape[0]):
                        self.misfitDiag += (wd[ii] * self.prob.F[ii, :])**2.


            diagA = self.misfitDiag + self.invProb.beta*regDiag

            PC = Utils.sdiag((diagA)**-1.)
            self.opt.approxHinv = PC

    def endIter(self):
        # Cool the threshold parameter
        if self.onlyOnStart is True:
            return

        if getattr(self.opt, 'approxHinv', None) is None:

            # Update the pre-conditioner
            if self.ComboRegFun:

                regDiag = []
                for reg in self.reg.objfcts:
                    regDiag.append((reg.W.T*reg.W).diagonal())

                regDiag = np.hstack(regDiag)

            else:

                regDiag = (self.reg.W.T*self.reg.W).diagonal()

#            if self.ComboMisfitFun:
#
#                misfitDiag = []
#                for misfit in self.dmisfit.objfcts:
#                    misfitDiag.append(np.sum(misfit.prob.F**2., axis=0))
#
#                misfitDiag = np.hstack(regDiag)
#
#            else:
#                misfitDiag = np.sum(self.dmisfit.prob.F**2., axis=0)

            diagA = self.misfitDiag + self.invProb.beta*regDiag

            PC = Utils.sdiag((diagA)**-1.)
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


class Amplitude_Inv_Iter(InversionDirective):
    """
    Directive to take care of re-weighting and pre-conditioning of
    the non-linear magnetic amplitude problem.

    """
    ptype = 'Amp'
    test = False
    mapping = None
    ComboObjFun = False

    def initialize(self):

        # Check if it is a ComboObjective
        if not isinstance(self.reg, Regularization.BaseComboRegularization):

            # It is a Combo objective, so will have to loop
            self.ComboObjFun = True

        self.reg.JtJdiag = self.getJtJdiag()

        if self.test:

            wr = np.sum(self.prob.F**2., axis=0)**0.5
            wr = wr / wr.max()

        else:
            wr = self.reg.JtJdiag**0.5
            wr = wr / wr.max()

        if self.ComboObjFun:
            for reg in self.reg.objfcts:
                reg.cell_weights = reg.mapping * wr
                reg.model = self.opt.xc
        else:
            self.reg.cell_weights = self.reg.mapping * wr

        if self.ptype == 'MVI-S':

            for reg in self.reg.objfcts[1:]:
                eps_a = self.reg.objfcts[0].eps_p
                norm_a = self.reg.objfcts[0].norms[0]
                f_m = self.reg.objfcts[0].objfcts[0].f_m
                max_a = np.max(eps_a**(1-norm_a/2.)*f_m /
                               (f_m**2. + eps_a**2.)**(1-norm_a/2.))

                eps_tp = reg.eps_q
                f_m = reg.objfcts[1].f_m
                norm_tp = reg.norms[1]
                max_tp = np.max(eps_tp**(1-norm_tp/2.)*f_m /
                                (f_m**2. + eps_tp**2.)**(1-norm_tp/2.))

                reg.scale = max_a/max_tp
                reg.cell_weights *= reg.scale

        # Update the pre-conditioner
        if self.ComboObjFun:

            reg_diag = []
            for reg in self.reg.objfcts:
                reg_diag.append(self.invProb.beta*(reg.W.T*reg.W).diagonal())

            diagA = self.reg.JtJdiag + np.hstack(reg_diag)

        else:
            diagA = self.reg.JtJdiag + self.invProb.beta*(self.reg.W.T*self.reg.W).diagonal()

        PC = Utils.sdiag((diagA)**-1.)
        self.opt.approxHinv = PC

        # if getattr(self.opt, 'approxHinv', None) is None:
        #     diagA = self.reg.JtJdiag + self.invProb.beta*(self.reg.W.T*self.reg.W).diagonal()
        #     PC = Utils.sdiag((self.prob.chiMap.deriv(None).T * diagA)**-1.)
        #     self.opt.approxHinv = PC

    def endIter(self):

        # Re-initialize the field derivatives
        if self.ptype == 'Amp':
            self.prob._dfdm = None
            self.prob.chi = self.invProb.model
        elif self.ptype == 'MVI-S':
            self.prob._S = None

        self.reg.JtJdiag = self.getJtJdiag()

        if not self.test:
            wr = self.reg.JtJdiag**0.5
            wr = wr / wr.max()

            if self.ComboObjFun:
                for reg in self.reg.objfcts:
                    reg.cell_weights = reg.mapping * wr

            else:
                self.reg.cell_weights = self.reg.mapping * wr

        if self.ptype == 'MVI-S':

            for reg in self.reg.objfcts[1:]:
                eps_a = self.reg.objfcts[0].eps_p
                norm_a = self.reg.objfcts[0].norms[0]
                f_m = self.reg.objfcts[0].objfcts[0].f_m
                max_a = np.max(eps_a**(1-norm_a/2.)*f_m /
                               (f_m**2. + eps_a**2.)**(1-norm_a/2.))

                eps_tp = reg.eps_q
                f_m = reg.objfcts[1].f_m
                norm_tp = reg.norms[1]
                max_tp = np.max(eps_tp**(1-norm_tp/2.)*f_m /
                                (f_m**2. + eps_tp**2.)**(1-norm_tp/2.))

                reg.scale = max_a/max_tp
                reg.cell_weights *= reg.scale

        if getattr(self.opt, 'approxHinv', None) is not None:
            # Update the pre-conditioner
            # Update the pre-conditioner
            if self.ComboObjFun:

                reg_diag = []
                for reg in self.reg.objfcts:
                    reg_diag.append(self.invProb.beta*(reg.W.T*reg.W).diagonal())

                diagA = self.reg.JtJdiag + np.hstack(reg_diag)

            else:
                diagA = self.reg.JtJdiag + self.invProb.beta*(self.reg.W.T*self.reg.W).diagonal()

            PC = Utils.sdiag(( diagA)**-1.)
            self.opt.approxHinv = PC

    def getJtJdiag(self):
        """
            Compute explicitely the main diagonal of JtJ for linear problem
        """
        nC = self.prob.chiMap.shape[0]
        nD = self.survey.nD

        JtJdiag = np.zeros(nC)

        if self.ptype == 'Amp':
            for ii in range(nC):

                JtJdiag[ii] = np.sum((self.prob.dfdm*self.prob.F[:, ii])**2.)

        elif self.ptype == 'MVI-S':

            for ii in range(nD):

                JtJdiag += (self.prob.F[ii, :] * self.prob.S)**2.

            JtJdiag += 1e-10

        return JtJdiag


class ProjSpherical(InversionDirective):

    def initialize(self):

        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        xyz = Magnetics.atp2xyz(x)
        m = Magnetics.xyz2atp(xyz)

        self.invProb.model = m
        self.prob.chi = m
        self.opt.xc = m

    def endIter(self):

        x = self.invProb.model
        # Convert to cartesian than back to avoid over rotation
        xyz = Magnetics.atp2xyz(x)
        m = Magnetics.xyz2atp(xyz)

        self.invProb.model = m
        self.invProb.phi_m_last = self.reg(m)
        self.prob.chi = m
        self.opt.xc = m
