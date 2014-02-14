from SimPEG import Solver
from SimPEG.Problem import BaseProblem
from simpegEM.Utils import Sources
from FieldsTDEM import FieldsTDEM
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc
import numpy as np


class MixinInitialFieldCalc(object):
    """docstring for MixinInitialFieldCalc"""

    def getInitialFields(self):
        if self.data.txType == 'VMD_MVP':
            # Vertical magnetic dipole, magnetic vector potential
            F = self._getInitialFields_VMD_MVP()
        else:
            exStr = 'Invalid txType: ' + str(self.data.txType)
            raise Exception(exStr)
        return F

    def _getInitialFields_VMD_MVP(self):
        if self.mesh._meshType is 'CYL1D':
            MVP = Sources.MagneticDipoleVectorPotential(np.r_[0,0,self.data.txLoc], np.c_[np.zeros(self.mesh.nN), self.mesh.gridN], 'x')
        elif self.mesh._meshType is 'TENSOR':
            MVPx = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEx, 'x')
            MVPy = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEy, 'y')
            MVPz = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEz, 'z')
            MVP = np.concatenate((MVPx, MVPy, MVPz))

        # Initialize field object
        F = FieldsTDEM(self.mesh, 1, self.times.size, 'b')

        # Set initial B
        F.b0 = self.mesh.edgeCurl*MVP

        return F

class MixinTimeStuff(object):
    """docstring for MixinTimeStuff"""

    def dt():
        doc = "Size of time steps"
        def fget(self):
            return self._dt
        def fdel(self):
            del self._dt
        return locals()
    dt = property(**dt())

    def nsteps():
        doc = "Number of steps to take"
        def fget(self):
            return self._nsteps
        def fdel(self):
            del self._nsteps
        return locals()
    nsteps = property(**nsteps())

    def times():
        doc = "Modelling times"
        def fget(self):
            t = np.r_[1:self.nsteps[0]+1]*self.dt[0]
            for i in range(1,self.dt.size):
                t = np.r_[t, np.r_[1:self.nsteps[i]+1]*self.dt[i]+t[-1]]
            return t
        return locals()
    times = property(**times())

    def getDt(self, tInd):
        return np.concatenate([self.dt[i].repeat(self.nsteps[i]) for i in range(self.dt.size)])[tInd]

    def setTimes(self, dt, nsteps):
        dt = np.array(dt)
        nsteps = np.array(nsteps)
        assert dt.size==nsteps.size, "dt, nsteps must be same length"
        self._dt = dt
        self._nsteps = nsteps

    @property
    def nTimes(self):
        return self.times.size


class ProblemBaseTDEM(MixinTimeStuff, MixinInitialFieldCalc, BaseProblem):
    """docstring for ProblemTDEM1D"""
    def __init__(self, mesh, model, **kwargs):
        BaseProblem.__init__(self, mesh, model, **kwargs)


    ####################################################
    # Physical Properties
    ####################################################

    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value
    _sigma = None

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MfMui(self): return self._MfMui

    @property
    def MeSigma(self): return self._MeSigma

    @property
    def MeSigmaI(self): return self._MeSigmaI

    def makeMassMatrices(self, m):
        self._MeSigma = self.mesh.getMass(m, loc='e')
        self._MeSigmaI = sdiag(1/self.MeSigma.diagonal())
        self._MfMui = self.mesh.getMass(1/mu_0, loc='f')


    def calcFields(self, sol, solType, tInd):

        if solType == 'b':
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b
            # Todo: implement non-zero js
        else:
            errStr = 'solType: ' + solType
            raise NotImplementedError(errStr)

        return {'b':b, 'e':e}

    solveOpts = {'factorize':True,'backend':'scipy'}

    def fields(self, m, useThisRhs=None, useThisCalcFields=None):
        RHS = useThisRhs or self.getRHS
        CalcFields = useThisCalcFields or self.calcFields

        self.makeMassMatrices(m)

        F = self.getInitialFields()
        #TODO: Split next code to forward and adjoint.
        # fields would call forward
        dtFact = None
        for tInd, t in enumerate(self.times):
            dt = self.getDt(tInd)
            if dt!=dtFact:
                dtFact = dt
                A = self.getA(tInd)
                # print 'Factoring...   (dt = ' + str(dt) + ')'
                Asolve = Solver(A, options=self.solveOpts)
                # print 'Done'
            rhs = RHS(tInd, F)
            sol = Asolve.solve(rhs)
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            newFields = CalcFields(sol, self.solType, tInd)
            F.update(newFields, tInd)
        return F
