import Utils, Survey, Problem, numpy as np, scipy.sparse as sp, gc
from Utils.SolverUtils import *
import DataMisfit
import Regularization


class BaseInvProblem(object):
    """BaseInvProblem(dmisfit, reg, opt)"""

    __metaclass__ = Utils.SimPEGMetaClass

    beta    = 1.0    #: Trade-off parameter

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    dmisfit = None   #: DataMisfit
    reg     = None   #: Regularization
    opt     = None   #: Optimization program

    deleteTheseOnModelUpdate = [] # List of strings, e.g. ['_MeSigma', '_MeSigmaI']

    @property
    def curModel(self):
        """
            Sets the current model, and removes dependent properties
        """
        return getattr(self, '_curModel', None)
    @curModel.setter
    def curModel(self, value):
        if value is self.curModel:
            return # it is the same!
        self._curModel = value
        for prop in self.deleteTheseOnModelUpdate:
            if hasattr(self, prop):
                delattr(self, prop)

    def __init__(self, dmisfit, reg, opt, **kwargs):
        Utils.setKwargs(self, **kwargs)
        assert isinstance(dmisfit, DataMisfit.BaseDataMisfit), 'dmisfit must be a DataMisfit class.'
        assert isinstance(reg, Regularization.BaseRegularization), 'reg must be a Regularization class.'
        self.dmisfit = dmisfit
        self.reg = reg
        self.opt = opt
        self.prob, self.survey = dmisfit.prob, dmisfit.survey
        #TODO: Remove: (and make iteration printers better!)
        self.opt.parent = self

    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug: print 'Calling InvProblem.startup'

        if self.reg.mref is None:
            print 'SimPEG.InvProblem will set Regularization.mref to m0.'
            self.reg.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.curModel = m0

        print 'SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv. \n    ***Done using direct methods***'
        self.opt.bfgsH0 = Solver(self.reg.eval2Deriv(self.curModel))

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        #TODO: check for warmstart
        self.curModel = m
        gc.collect()

        u = self.prob.fields(m)

        phi_d = self.dmisfit.eval(m, u=u)
        phi_m = self.reg.eval(m)

        self.dpred = self.survey.dpred(m, u=u)  # This is a cheap matrix vector calculation.

        self.phi_d, self.phi_d_last  = phi_d, self.phi_d
        self.phi_m, self.phi_m_last  = phi_m, self.phi_m

        f = phi_d + self.beta * phi_m

        out = (f,)
        if return_g:
            phi_dDeriv = self.dmisfit.evalDeriv(m, u=u)
            phi_mDeriv = self.reg.evalDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.eval2Deriv(m, v, u=u)
                phi_m2Deriv = self.reg.eval2Deriv(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (H,)
        return out if len(out) > 1 else out[0]
