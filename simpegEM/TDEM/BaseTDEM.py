from SimPEG import Solver
from SimPEG.Problem import BaseProblem
from simpegEM.Utils import Sources
from FieldsTDEM import FieldsTDEM
from scipy.constants import mu_0
from SimPEG.Utils import sdiag, mkvc
import numpy as np


class MixinInitialFieldCalc(object):
    """docstring for MixinInitialFieldCalc"""

    storeTheseFields = 'b'

    def getInitialFields(self):
        if self.survey.txType == 'VMD_MVP':
            # Vertical magnetic dipole, magnetic vector potential
            F = self._getInitialFields_VMD_MVP()
        else:
            exStr = 'Invalid txType: ' + str(self.survey.txType)
            raise Exception(exStr)
        return F

    def _getInitialFields_VMD_MVP(self):
        if self.mesh._meshType is 'CYL':
            if self.mesh.isSymmetric:
                MVP = Sources.MagneticDipoleVectorPotential(self.survey.txLoc, self.mesh.gridEy, 'y')
                # MVP = Sources.MagneticDipoleVectorPotential(self.survey.txLoc, np.c_[np.zeros(self.mesh.nN), self.mesh.gridN], 'x')
            else:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
        elif self.mesh._meshType is 'TENSOR':
            MVPx = Sources.MagneticDipoleVectorPotential(self.survey.txLoc, self.mesh.gridEx, 'x')
            MVPy = Sources.MagneticDipoleVectorPotential(self.survey.txLoc, self.mesh.gridEy, 'y')
            MVPz = Sources.MagneticDipoleVectorPotential(self.survey.txLoc, self.mesh.gridEz, 'z')
            MVP = np.concatenate((MVPx, MVPy, MVPz))
        else:
            raise Exception('Unknown mesh for VMD')

        # Initialize field object
        F = FieldsTDEM(self.mesh, 1, self.times.size, store=self.storeTheseFields)

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
        doc = "Modeling times"
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
    def __init__(self, model, **kwargs):
        BaseProblem.__init__(self, model, **kwargs)


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
        sig = self.model.transform(m)
        self._MeSigma = self.mesh.getEdgeInnerProduct(sig)
        self._MeSigmaI = sdiag(1/self.MeSigma.diagonal())
        self._MfMui = self.mesh.getFaceInnerProduct(1/mu_0)


    def calcFields(self, sol, solType, tInd):

        if solType == 'b':
            b = sol
            e = self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui*b
            # Todo: implement non-zero js
        else:
            errStr = 'solType: ' + solType
            raise NotImplementedError(errStr)

        return {'b':b, 'e':e}

    Solver = Solver
    solveOpts = {}

    def fields(self, m):
        self.makeMassMatrices(m)
        F = self.getInitialFields()
        return self.forward(m, self.getRHS, self.calcFields, F=F)


    def forward(self, m, RHS, CalcFields, F=None):
        if F is None:
            F = FieldsTDEM(self.mesh, self.survey.nTx, self.nTimes, store=self.storeTheseFields)

        dtFact = None
        for tInd, t in enumerate(self.times):
            dt = self.getDt(tInd)
            if dt!=dtFact:
                dtFact = dt
                A = self.getA(tInd)
                # print 'Factoring...   (dt = ' + str(dt) + ')'
                Asolve = self.Solver(A, **self.solveOpts)
                # print 'Done'
            rhs = RHS(tInd, F)
            sol = Asolve.solve(rhs)
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            newFields = CalcFields(sol, self.solType, tInd)
            F.update(newFields, tInd)
        return F

    def adjoint(self, m, RHS, CalcFields, F=None):
        if F is None:
            F = FieldsTDEM(self.mesh, self.survey.nTx, self.nTimes, store=self.storeTheseFields)

        dtFact = None
        for tInd, t in reversed(list(enumerate(self.times))):
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

