from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
import numpy as np
import properties
from SimPEG.Utils import sdiag


class BaseRx(SimPEG.Survey.BaseTimeRx):
    locs = None
    rxType = None

    data_type = properties.StringChoice(
        "Type of DC-IP survey",
        required=True,
        default="volt",
        choices=[
           "volt",
           "apparent_resistivity",
           "apparent_chargeability"
        ]
    )

    data_type = 'volt'

    knownRxTypes = {
                    'phi': ['phi', None],
                    'ex': ['e', 'x'],
                    'ey': ['e', 'y'],
                    'ez': ['e', 'z'],
                    'jx': ['j', 'x'],
                    'jy': ['j', 'y'],
                    'jz': ['j', 'z'],
                    }

    def __init__(self, locs, times, rxType, **kwargs):
        SimPEG.Survey.BaseTimeRx.__init__(self, locs, times, rxType, **kwargs)

    @property
    def dc_voltage(self):
        return self._dc_voltage

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

    def eval(self, src, mesh, f):
        P = self.getP(mesh, self.projGLoc(f))
        return P*f[src, self.projField]

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P*v
        elif adjoint:
            return P.T*v


class Dipole(BaseRx):
    """
    Dipole receiver
    """

    def __init__(self, locsM, locsN, times, rxType='phi', **kwargs):
        assert locsM.shape == locsN.shape, (
            'locsM and locsN need to be the same size'
        )
        locs = [np.atleast_2d(locsM), np.atleast_2d(locsN)]
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

        P0 = mesh.getInterpolationMat(self.locs[0], Gloc)
        P1 = mesh.getInterpolationMat(self.locs[1], Gloc)
        P = P0 - P1

        if self.data_type == 'apparent_resistivity':
            P = sdiag(1./self.geometric_factor) * P
        elif self.data_type == 'apparent_chargeability':
            P = sdiag(1./self.dc_voltage) * P

        if self.storeProjections:
            self._Ps[mesh] = P

        return P


class Pole(BaseRx):
    """
    Pole receiver
    """

    def __init__(self, locsM, times, rxType='phi', **kwargs):
        locs = np.atleast_2d(locsM)
        # We may not need this ...
        BaseRx.__init__(self, locs, times, rxType)

    @property
    def nD(self):
        """Number of data in the receiver."""
        # return self.locs[0].shape[0] * len(self.times)
        return self.locs.shape[0]

    @property
    def nRx(self):
        """Number of data in the receiver."""
        return self.locs.shape[0]

        # Not sure why ...
        # return int(self.locs[0].size / 2)

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P = mesh.getInterpolationMat(self.locs, Gloc)

        if self.data_type == 'apparent_resistivity':
            P = sdiag(1./self.geometric_factor) * P
        elif self.data_type == 'apparent_chargeability':
            P = sdiag(1./self.dc_voltage) * P

        if self.storeProjections:
            self._Ps[mesh] = P

        return P
