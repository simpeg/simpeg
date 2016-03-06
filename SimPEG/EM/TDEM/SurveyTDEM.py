import SimPEG
from SimPEG import np, Utils
from SimPEG.Utils import Zero, Identity 
from scipy.constants import mu_0 
from SimPEG.EM.Utils import * 

####################################################
# Receivers
####################################################

class Rx(SimPEG.Survey.BaseTimeRx):

    knownRxTypes = {
                    'ex':['e', 'Ex', 'N'],
                    'ey':['e', 'Ey', 'N'],
                    'ez':['e', 'Ez', 'N'],

                    'bx':['b', 'Fx', 'N'],
                    'by':['b', 'Fy', 'N'],
                    'bz':['b', 'Fz', 'N'],

                    'dbxdt':['b', 'Fx', 'CC'],
                    'dbydt':['b', 'Fy', 'CC'],
                    'dbzdt':['b', 'Fz', 'CC'],
                   }

    def __init__(self, locs, times, rxType):
        SimPEG.Survey.BaseTimeRx.__init__(self, locs, times, rxType)

    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    @property
    def projGLoc(self):
        """Grid Location projection (e.g. Ex Fy ...)"""
        return self.knownRxTypes[self.rxType][1]

    @property
    def projTLoc(self):
        """Time Location projection (e.g. CC N)"""
        return self.knownRxTypes[self.rxType][2]

    def getTimeP(self, timeMesh):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        if self.rxType in ['dbxdt','dbydt','dbzdt']:
            return timeMesh.getInterpolationMat(self.times, self.projTLoc)*timeMesh.faceDiv
        else:
            return timeMesh.getInterpolationMat(self.times, self.projTLoc)

    def eval(self, src, mesh, timeMesh, u):
        P = self.getP(mesh, timeMesh)
        u_part = Utils.mkvc(u[src, self.projField, :])
        return P*u_part

    def evalDeriv(self, src, mesh, timeMesh, df_dm, adjoint=False):
        P = self.getP(mesh, timeMesh)

        if not adjoint:
            return P * Utils.mkvc(df_dm[src, self.projField+'Deriv', :])
        elif adjoint:
            return P.T * df_dm[src, self]

####################################################
# Sources
####################################################

class BaseWaveform(object):

    def __init__(self, offTime=0., hasInitialFields=False):
        self.offTime = offTime
        self.hasInitialFields = hasInitialFields

    def _assertMatchesPair(self, pair):
        assert (isinstance(self, pair)
            ), "Waveform object must be an instance of a %s BaseWaveform class."%(pair.__name__)

    def eval(self, time):
        raise NotImplementedError 

    def evalDeriv(self, time):
        raise NotImplementedError # needed for E-formulation 


class StepOffWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        return 0.


class RawWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        raise NotImplementedError('RawWaveform has not been implemented, you should write it!')


class TriangularWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        raise NotImplementedError('TriangularWaveform has not been implemented, you should write it!')



class BaseSrc(SimPEG.Survey.BaseSrc):

    rxPair = Rx
    integrate = True
    waveformPair = BaseWaveform

    @property
    def waveform(self):
        "A waveform instance is not None"
        return getattr(self, '_waveform', None)
    @waveform.setter
    def waveform(self, val):
        # if self.waveform is None:
        val._assertMatchesPair(self.waveformPair)
        self._waveform = val


    def __init__(self, rxList, waveform = None ):
        self.waveform = waveform or StepOffWaveform()
        SimPEG.Survey.BaseSrc.__init__(self, rxList) 


    def bInitial(self, prob):
        return Zero()

    def bInitialDeriv(self, prob, v=None, adjoint=False):
        return Zero()

    def eval(self, prob, time):
        S_m = self.S_m(prob, time)
        S_e = self.S_e(prob, time)
        return S_m, S_e 

    def evalDeriv(self, prob, time, v=None, adjoint=False):
        if v is not None:
            return self.S_mDeriv(prob, time, v, adjoint), self.S_eDeriv(prob, time, v, adjoint)
        else:
            return lambda v: self.S_mDeriv(prob, time, v, adjoint), lambda v: self.S_eDeriv(prob, time, v, adjoint)

    def S_m(self, prob, time):
        return Zero()

    def S_e(self, prob, time):
        return Zero()

    def S_mDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()

    def S_eDeriv(self, prob, time, v=None, adjoint=False):
        return Zero()


class MagDipole(BaseSrc):
    def __init__(self, rxList, waveform=None, loc=None, orientation='Z', moment=1., mu=mu_0):  

        self.loc = loc
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.moment = moment
        self.mu = mu
        self.integrate = False
        BaseSrc.__init__(self, rxList, waveform)

    def _bfromVectorPotential(self, prob):
        if prob._eqLocs is 'FE':
            gridX = prob.mesh.gridEx
            gridY = prob.mesh.gridEy
            gridZ = prob.mesh.gridEz
            C = prob.mesh.edgeCurl

        elif prob._eqLocs is 'EF':
            gridX = prob.mesh.gridFx
            gridY = prob.mesh.gridFy
            gridZ = prob.mesh.gridFz
            C = prob.mesh.edgeCurl.T


        if prob.mesh._meshType is 'CYL':
            if not prob.mesh.isSymmetric:
                raise NotImplementedError('Non-symmetric cyl mesh not implemented yet!')
            a = MagneticDipoleVectorPotential(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)

        else:
            srcfct = MagneticDipoleVectorPotential
            ax = srcfct(self.loc, gridX, 'x', mu=self.mu, moment=self.moment)
            ay = srcfct(self.loc, gridY, 'y', mu=self.mu, moment=self.moment)
            az = srcfct(self.loc, gridZ, 'z', mu=self.mu, moment=self.moment)
            a = np.concatenate((ax, ay, az))

        return C*a


    def bInitial(self, prob):
        eqLocs = prob._eqLocs

        if self.waveform.hasInitialFields is False:
            return Zero()

        return self._bfromVectorPotential(prob)

    def S_m(self, prob, time):
        if self.waveform.hasInitialFields is False:
            raise NotImplementedError
        return Zero()

    def S_e(self, prob, time):
        if self.waveform.hasInitialFields is False:
            raise NotImplementedError
        return Zero()

####################################################
# Survey
####################################################

class Survey(SimPEG.Survey.BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    srcPair = BaseSrc
    rxPair = Rx

    def __init__(self, srcList, **kwargs):
        # Sort these by frequency
        self.srcList = srcList
        SimPEG.Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, u):
        data = SimPEG.Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, self.mesh, self.prob.timeMesh, u)
        return data

    def evalDeriv(self, u, v=None, adjoint=False):
        assert v is not None, 'v to multiply must be provided.'

        if not adjoint:
            data = SimPEG.Survey.Data(self)
            for src in self.srcList:
                for rx in src.rxList:
                    data[src, rx] = rx.evalDeriv(src, self.mesh, self.prob.timeMesh, u, v)
            return data
        else:
            f = FieldsTDEM(self.mesh, self)
            for src in self.srcList:
                for rx in src.rxList:
                    Ptv = rx.evalDeriv(src, self.mesh, self.prob.timeMesh, u, v, adjoint=True)
                    Ptv = Ptv.reshape((-1, self.prob.timeMesh.nN), order='F')
                    if rx.projField not in f: # first time we are projecting
                        f[src, rx.projField, :] = Ptv
                    else: # there are already fields, so let's add to them!
                        f[src, rx.projField, :] += Ptv
            return f


