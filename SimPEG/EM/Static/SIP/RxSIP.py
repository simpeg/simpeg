from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
import numpy as np
from SimPEG.Utils import Zero, closestPoints

class BaseRx(SimPEG.Survey.BaseTimeRx):
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

    def __init__(self, locs, times, rxType, **kwargs):
        SimPEG.Survey.BaseTimeRx.__init__(self, locs, times, rxType, **kwargs)

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

    def getTimeP(self, timesall):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        time_inds = np.in1d(timesall, self.times)
        return time_inds

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P*v
        elif adjoint:
            return P.T*v


# DC.Rx.Dipole(locs)
class Dipole(BaseRx):

    rxgeom = "dipole"

    def __init__(self, locsM, locsN, times, rxType = 'phi', **kwargs):
        assert locsM.shape == locsN.shape, 'locsM and locsN need to be the same size'
        if np.array_equal(locsM, locsN):
            self.rxgeom = "pole"
            locs = [locsM]
        else:
            locs = [locsM, locsN]
        # We may not need this ...
        BaseRx.__init__(self, locs, times, rxType)

    @property
    def nD(self):
        """Number of data in the receiver."""
        # return self.locs[0].shape[0] * len(self.times)
        return self.locs[0].shape[0]

    @property
    def nRx(self):
        """Number of data in the receiver."""
        return self.locs[0].shape[0]

        # Not sure why ...
        # return int(self.locs[0].size / 2)


    def getP(self, mesh, Gloc):

        if mesh in self._Ps:
            return self._Ps[mesh]

        if self.rxgeom == "dipole":
            P0 = mesh.getInterpolationMat(self.locs[0], Gloc)
            P1 = mesh.getInterpolationMat(self.locs[1], Gloc)
            P = P0 - P1
        elif self.rxgeom == "pole":
            P = mesh.getInterpolationMat(self.locs[0], Gloc)

        if self.storeProjections:
            self._Ps[mesh] = P

        return P
