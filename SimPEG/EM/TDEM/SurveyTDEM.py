from SimPEG import Utils, Survey, np
from SimPEG.Survey import BaseSurvey

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



