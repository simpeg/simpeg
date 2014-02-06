from SimPEG import Utils
from SimPEG.Data import BaseData
from SimPEG.Problem import BaseProblem
from simpegEM.Utils import Sources
import numpy as np


class DataTDEM1D(BaseData):
    """
        docstring for DataTDEM1D
    """

    txLoc = None #: txLoc
    txType = None #: txType
    rxLoc = None #: rxLoc
    rxType = None #: rxType
    timeCh = None #: timeCh

    def __init__(self, **kwargs):
        BaseData.__init__(self, **kwargs)
        Utils.setKwargs(self, **kwargs)

class MixinInitialFieldCalc(object):
    """docstring for MixinInitialFieldCalc"""
                        
    def getInitialFields(self):
        if self.data.txType == 'VMD_MVP':
            # Vertical magnetic dipole, magnetic vector potential
            self._getInitialFields_VMD_MVP()
        else:
            exStr = 'Invalid txType: ' + str(self.data.txType)
            raise Exception(exStr)
        pass

    def _getInitialFields_VMD_MVP(self):
        print 'VMD INI'
        if self.mesh._meshType is 'CYL1D':
            MVP = Sources.MagneticDipoleVectorPotential(np.r_[0,0,self.data.txLoc], np.c_[np.zeros(self.mesh.nN), self.mesh.gridN], 'x')
        elif self.mesh._meshType is 'TENSOR':
            MVPx = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEx, 'x')
            MVPy = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEy, 'y')
            MVPz = Sources.MagneticDipoleVectorPotential(self.data.txLoc, self.mesh.gridEz, 'z')
            MVP = np.concatenate((MVPx, MVPy, MVPz))
        
        # Initialize field object
        self.F = FieldsTDEM(self.mesh, self.tCalc.size, 'b')

        # Set initial B
        self.F.b0 = self.mesh.edgeCurl*MVP

class ProblemTDEM1D(MixinInitialFieldCalc, BaseProblem):
    """docstring for ProblemTDEM1D"""
    def __init__(self, mesh, model, **kwargs):
        BaseProblem.__init__(self, mesh, model, **kwargs)
        

    # Time stuff ########################################

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

    def tCalc():
        doc = "Modelling times"
        def fget(self):
            t = np.r_[1:self.nsteps[0]+1]*self.dt[0]
            for i in range(1,self.dt.size):
                t = np.r_[t, np.r_[1:self.nsteps[i]+1]*self.dt[i]+t[-1]]
            return t
        return locals()
    tCalc = property(**tCalc())

    def getDt(self, tInd):
        return np.concatenate([self.dt[i].repeat(self.nsteps[i]) for i in range(self.dt.size)])[tInd]

    def setTimes(self, dt, nsteps):
        dt = np.array(dt)
        nsteps = np.array(nsteps)
        assert dt.size==nsteps.size, "dt, nsteps must be same length"
        self._dt = dt
        self._nsteps = nsteps

    # End Time stuff ########################################

    def field(self, m):
        F = self.getInitialFields()
        Asolve = None
        for i, dt in enumerate(self.times):
            A = self.getA(i, F)
            rhs = self.getRHS(i, F)
            if Asolve is None or redoSolver:
                Asolve = Solver(A,options=self.solveOpts)
            sol = Asolve.Solve(rhs)
            self.updateField(sol, F)

        return F



class FieldsTDEM(object):

    phi0 = None #: Initial electric potential
    A0 = None #: Initial magnetic vector potential
    e0 = None #: Initial electric field
    b0 = None #: Initial magnetic flux density
    j0 = None #: Initial current density
    h0 = None #: Initial magnetic field

    phi = None #: Electric potential
    A = None #: Magnetic vector potential
    e = None #: Electric field
    b = None #: Magnetic flux density
    j = None #: Current density
    h = None #: Magnetic field

    """docstring for FieldsTDEM"""
    def __init__(self, mesh, nTimes, store):
        
        self.nTimes = nTimes
        self.nC = mesh.nC
        self.nE = mesh.nE
        self.nF = mesh.nF
        self.nN = mesh.nN


