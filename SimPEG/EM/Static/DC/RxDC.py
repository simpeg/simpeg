import SimPEG
# from SimPEG.EM.Base import BaseEMSurvey
from SimPEG.Utils import Zero, closestPoints

class BaseRx(SimPEG.Survey.BaseRx):
    locs = None
    rxType = None

    knownRxTypes = {
                    'phi':['phi',None],
                    'ex':['e','x'],
                    'ey':['e','y'],
                    'ez':['e','z'],
                    'jx':['j','x'],
                    'jy':['j','y'],
                    'jz':['j','z'],
                    }

    def __init__(self, locs, rxType, **kwargs):
        SimPEG.Survey.BaseRx.__init__(self, locs, rxType, **kwargs)


    @property
    def projField(self):
        """Field Type projection (e.g. e b ...)"""
        return self.knownRxTypes[self.rxType][0]

    def projGLoc(self, f):
        """Grid Location projection (e.g. Ex Fy ...)"""
        comp = self.knownRxTypes[self.rxType][1]
        if comp is not None:
            return f._GLoc(self.rxType) + comp
        return f._GLoc(self.rxType)

    def eval(self, src, mesh, f):
        P = self.getP(mesh, self.projGLoc(f))
        return P*f[src, self.projField]

# DC.Rx.Dipole(locs)
class Dipole(BaseRx):

    def __init__(self, locsM, locsN, rxType = 'phi', **kwargs):
        assert locsM.shape == locsN.shape, 'locsM and locsN need to be the same size'
        locs = [locsM, locsN]
        # We may not need this ...
        BaseRx.__init__(self, locs, rxType)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return int(self.locs[0].size / 2)

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P0 = mesh.getInterpolationMat(self.locs[0], Gloc)
        P1 = mesh.getInterpolationMat(self.locs[1], Gloc)
        P = P0 - P1

        if self.storeProjections:
            self._Ps[mesh] = P

        return P


# class Pole(BaseRx):


