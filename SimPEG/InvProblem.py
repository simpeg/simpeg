import Utils, Survey, Problem, numpy as np, scipy.sparse as sp, gc
from Utils.SolverUtils import *
from DataMisfit import _splitForward

class BaseInvProblem(object):
    """BaseInvProblem(forward, reg, **kwargs)"""

    __metaclass__ = Utils.SimPEGMetaClass

    beta    = 1.0    #: Trade-off parameter

    debug   = False  #: Print debugging information
    counter = None   #: Set this to a SimPEG.Utils.Counter() if you want to count things

    reg     = None   #: Regularization
    dmisfit = None   #: DataMisfit
    opt     = None   #: Optimization program

    u_current = None #: The most current evaluated field
    m_current = None #: The most current model


    def __init__(self, forward, reg, dmisfit, opt, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.prob, self.survey = _splitForward(forward)
        self.reg = reg
        self.dmisfit = dmisfit
        self.opt = opt

    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug: print 'Calling InvProblem.startup'

        if self.reg.mref is None:
            print 'Regularization has not set mref. SimPEG.InvProblem will set it to m0.'
            self.reg.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.m_current = m0

        print 'Setting bfgsH0 to the inverse of the modelObj2Deriv. Done using direct methods.'
        self.opt.bfgsH0 = Solver(self.reg.modelObj2Deriv(self.m_current))

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        self.u_current = None
        self.m_current = m
        forward = self.prob
        gc.collect()

        u = self.prob.fields(m)
        self.u_current = u

        phi_d = self.dmisfit.dataObj(forward, m, u=u)
        phi_m = self.reg.modelObj(m)

        self.dpred = self.survey.dpred(m, u=u)  # This is a cheap matrix vector calculation.

        self.phi_d, self.phi_d_last  = phi_d, self.phi_d
        self.phi_m, self.phi_m_last  = phi_m, self.phi_m

        f = phi_d + self.beta * phi_m

        out = (f,)
        if return_g:
            phi_dDeriv = self.dmisfit.dataObjDeriv(forward, m, u=u)
            phi_mDeriv = self.reg.modelObjDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.dataObj2Deriv(forward, m, v, u=u)
                phi_m2Deriv = self.reg.modelObj2Deriv(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (H,)
        return out if len(out) > 1 else out[0]
