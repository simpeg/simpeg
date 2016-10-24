from __future__ import division, print_function
import SimPEG
from SimPEG import np, Utils
from SimPEG.Utils import Zero, Identity
from scipy.constants import mu_0
from SimPEG.EM.Utils import *
from . import SrcTDEM as Src


####################################################
# Receivers
####################################################

class Rx(SimPEG.Survey.BaseTimeRx):

    knownRxTypes = {
                    'ex': ['e', 'Ex', 'N'],
                    'ey': ['e', 'Ey', 'N'],
                    'ez': ['e', 'Ez', 'N'],

                    'bx': ['b', 'Fx', 'N'],
                    'by': ['b', 'Fy', 'N'],
                    'bz': ['b', 'Fz', 'N'],

                    'dbxdt': ['b', 'Fx', 'CC'],
                    'dbydt': ['b', 'Fy', 'CC'],
                    'dbzdt': ['b', 'Fz', 'CC'],
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
        if self.rxType in ['dbxdt', 'dbydt', 'dbzdt']:
            return timeMesh.getInterpolationMat(self.times,
                                                self.projTLoc)*timeMesh.faceDiv
        else:
            return timeMesh.getInterpolationMat(self.times, self.projTLoc)

    def eval(self, src, mesh, timeMesh, u):
        P = self.getP(mesh, timeMesh)
        u_part = Utils.mkvc(u[src, self.projField, :])
        return P*u_part

    def evalDeriv(self, src, mesh, timeMesh, v, adjoint=False):
        P = self.getP(mesh, timeMesh)
        if not adjoint:
            return P * v # Utils.mkvc(v[src, self.projField+'Deriv', :])
        elif adjoint:
            # dP_dF_T = P.T * v #[src, self]
            # newshape = (len(dP_dF_T)/timeMesh.nN, timeMesh.nN )
            return P.T * v # np.reshape(dP_dF_T, newshape, order='F')


####################################################
# Survey
####################################################

class Survey(SimPEG.Survey.BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    srcPair = Src.BaseSrc
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
        raise Exception('Use Receivers to project fields deriv.')


