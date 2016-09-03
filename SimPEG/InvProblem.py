from __future__ import print_function
from . import Utils
from . import Survey
from . import Problem
import numpy as np
import scipy.sparse as sp
import gc
from .Utils.SolverUtils import *
from . import DataMisfit
from . import Regularization


class BaseInvProblem(object):
    """BaseInvProblem(dmisfit, reg, opt)"""

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
        self.reg.parent = self
        self.dmisfit.parent = self

    @Utils.callHooks('startup')
    def startup(self, m0):
        """startup(m0)

            Called when inversion is first starting.
        """
        if self.debug: print('Calling InvProblem.startup')

        if self.reg.mref is None:
            print('SimPEG.InvProblem will set Regularization.mref to m0.')
            self.reg.mref = m0

        self.phi_d = np.nan
        self.phi_m = np.nan

        self.curModel = m0

        print("""SimPEG.InvProblem is setting bfgsH0 to the inverse of the eval2Deriv.
                    ***Done using same Solver and solverOpts as the problem***""")
        self.opt.bfgsH0 = self.prob.Solver(self.reg.eval2Deriv(self.curModel), **self.prob.solverOpts)

    @property
    def warmstart(self):
        return getattr(self, '_warmstart', [])
    @warmstart.setter
    def warmstart(self, value):
        assert type(value) is list, 'warmstart must be a list.'
        for v in value:
            assert type(v) is tuple, 'warmstart must be a list of tuples (m, u).'
            assert len(v) == 2, 'warmstart must be a list of tuples (m, u). YOURS IS NOT LENGTH 2!'
            assert isinstance(v[0], np.ndarray), 'first warmstart value must be a model.'
        self._warmstart = value

    def getFields(self, m, store=False, deleteWarmstart=True):
        f = None

        for mtest, u_ofmtest in self.warmstart:
            if m is mtest:
                f = u_ofmtest
                if self.debug: print('InvProb is Warm Starting!')
                break

        if f is None:
            f = self.prob.fields(m)

        if deleteWarmstart:
            self.warmstart = []
        if store:
            self.warmstart += [(m,f)]

        return f

    @Utils.timeIt
    def evalFunction(self, m, return_g=True, return_H=True):
        """evalFunction(m, return_g=True, return_H=True)
        """

        self.curModel = m
        gc.collect()

        # Store fields if doing a line-search
        f = self.getFields(m, store=(return_g==False and return_H==False))

        phi_d = self.dmisfit.eval(m, f=f)
        phi_m = self.reg.eval(m)

        self.dpred = self.survey.dpred(m, f=f)  # This is a cheap matrix vector calculation.

        self.phi_d, self.phi_d_last  = phi_d, self.phi_d
        self.phi_m, self.phi_m_last  = phi_m, self.phi_m

        phi = phi_d + self.beta * phi_m

        out = (phi,)
        if return_g:
            phi_dDeriv = self.dmisfit.evalDeriv(m, f=f)
            phi_mDeriv = self.reg.evalDeriv(m)

            g = phi_dDeriv + self.beta * phi_mDeriv
            out += (g,)

        if return_H:
            def H_fun(v):
                phi_d2Deriv = self.dmisfit.eval2Deriv(m, v, f=f)
                phi_m2Deriv = self.reg.eval2Deriv(m, v=v)

                return phi_d2Deriv + self.beta * phi_m2Deriv

            H = sp.linalg.LinearOperator( (m.size, m.size), H_fun, dtype=m.dtype )
            out += (H,)
        return out if len(out) > 1 else out[0]
