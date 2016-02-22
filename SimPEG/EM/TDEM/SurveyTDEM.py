from SimPEG import Survey, np
from SimPEG.Survey import BaseSurvey
from SimPEG.Utils import Zero, Identity 
from scipy.constants import mu_0 

####################################################
# Receivers
####################################################

class Rx(Survey.BaseTimeRx):

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
        Survey.BaseTimeRx.__init__(self, locs, times, rxType)

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

    def evalDeriv(self, src, mesh, timeMesh, u, v, adjoint=False):
        P = self.getP(mesh, timeMesh)

        if not adjoint:
            return P * Utils.mkvc(v[src, self.projField, :])
        elif adjoint:
            return P.T * v[src, self]

####################################################
# Sources
####################################################

class BaseWaveform(object):

    def __init__(self, offTime=0., hasInitialFields=False):
        self.offTime = offTime
        self.hasInitialFields = hasInitialFields

    def eval(self, time):
        raise NotImplementedError 


class StepOffWaveform(BaseWaveform):

    def __init__(self, offTime=0.):
        BaseWaveform.__init__(self, offTime, hasInitialFields=True)

    def eval(self, time):
        return 0.



class BaseSrc(Survey.BaseSrc):

    rxPair = Rx
    integrate = True
    waveformPair = StepOffWaveform

    @property
    def waveform(self):
        "A waveform instance is not None"
        return getattr(self, '_waveform', None)
    @waveform.setter
    def waveform(self, val):
        if self.waveform is None:
            val._assertMatchesPair(self.waveformPair)
            self._waveform = val


    def __init__(self, rxList, waveform = None):
        self.waveform = waveform 
        Survey.BaseSrc.__init__(self, rxList) 


    def bInitial(self, mesh):
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


class MagDipole(BaseSrc):
    def __init__(self, rxList, waveform, loc, orientation='Z', moment=1., mu=mu_0, **kwargs):  

        self.loc = loc
        self.orientation = orientation
        assert orientation in ['X','Y','Z'], "Orientation (right now) doesn't actually do anything! The methods in SrcUtils should take care of this..."
        self.moment = moment
        self.mu = mu
        self.integrate = False
        BaseSrc.__init__(self, rxList)






