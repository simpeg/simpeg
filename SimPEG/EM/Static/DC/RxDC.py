from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import SimPEG
import numpy as np
from SimPEG.Utils import closestPoints


class BaseRx(SimPEG.Survey.BaseRx):
    """
    Base DC receiver
    """
    locs = None
    rxType = None

    knownRxTypes = {
        'phi': ['phi', None],
        'ex': ['e', 'x'],
        'ey': ['e', 'y'],
        'ez': ['e', 'z'],
        'jx': ['j', 'x'],
        'jy': ['j', 'y'],
        'jz': ['j', 'z'],
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

    def evalDeriv(self, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P*v
        elif adjoint:
            return P.T*v


# DC.Rx.Dipole(locs)
class Dipole(BaseRx):
    """
    Dipole receiver
    """

    def __init__(self, locsM, locsN, rxType='phi', **kwargs):
        assert locsM.shape == locsN.shape, ('locsM and locsN need to be the '
                                            'same size')
        locs = [np.atleast_2d(locsM), np.atleast_2d(locsN)]
        # We may not need this ...
        BaseRx.__init__(self, locs, rxType)

    @property
    def nD(self):
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

        if self.storeProjections:
            self._Ps[mesh] = P

        return P


class Dipole_ky(BaseRx):
    """
    Dipole receiver for 2.5D simulations
    """

    def __init__(self, locsM, locsN, rxType='phi', **kwargs):
        assert locsM.shape == locsN.shape, ('locsM and locsN need to be the '
                                            'same size')
        locs = [np.atleast_2d(locsM), np.atleast_2d(locsN)]
        # We may not need this ...
        BaseRx.__init__(self, locs, rxType)

    @property
    def nD(self):
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
        if self.storeProjections:
            self._Ps[mesh] = P
        return P

    def eval(self, kys, src, mesh, f):
        P = self.getP(mesh, self.projGLoc(f))
        Pf = P*f[src, self.projField, :]
        return self.IntTrapezoidal(kys, Pf, y=0.)

    def evalDeriv(self, ky, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P*v
        elif adjoint:
            return P.T*v

    def IntTrapezoidal(self, kys, Pf, y=0.):
        phi = np.zeros(Pf.shape[0])
        nky = kys.size
        dky = np.diff(kys)
        dky = np.r_[dky[0], dky]
        phi0 = 1./np.pi*Pf[:, 0]
        for iky in range(nky):
            phi1 = 1./np.pi*Pf[:, iky]
            phi += phi1*dky[iky]/2.*np.cos(kys[iky]*y)
            phi += phi0*dky[iky]/2.*np.cos(kys[iky]*y)
            phi0 = phi1.copy()
        return phi


class Pole(BaseRx):
    """
    Pole receiver
    """

    def __init__(self, locsM, rxType='phi', **kwargs):

        locs = np.atleast_2d(locsM)
        # We may not need this ...
        BaseRx.__init__(self, locs, rxType)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P = mesh.getInterpolationMat(self.locs, Gloc)

        if self.storeProjections:
            self._Ps[mesh] = P

        return P


class Pole_ky(BaseRx):
    """
    Pole receiver for 2.5D simulations
    """

    def __init__(self, locsM, rxType='phi', **kwargs):

        locs = np.atleast_2d(locsM)
        # We may not need this ...
        BaseRx.__init__(self, locs, rxType)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh, Gloc):
        if mesh in self._Ps:
            return self._Ps[mesh]

        P = mesh.getInterpolationMat(self.locs, Gloc)

        if self.storeProjections:
            self._Ps[mesh] = P

        return P

    def eval(self, kys, src, mesh, f):
        P = self.getP(mesh, self.projGLoc(f))
        Pf = P*f[src, self.projField, :]
        return self.IntTrapezoidal(kys, Pf, y=0.)

    def evalDeriv(self, ky, src, mesh, f, v, adjoint=False):
        P = self.getP(mesh, self.projGLoc(f))
        if not adjoint:
            return P*v
        elif adjoint:
            return P.T*v

    def IntTrapezoidal(self, kys, Pf, y=0.):
        phi = np.zeros(Pf.shape[0])
        nky = kys.size
        dky = np.diff(kys)
        dky = np.r_[dky[0], dky]
        phi0 = 1./np.pi*Pf[:, 0]
        for iky in range(nky):
            phi1 = 1./np.pi*Pf[:, iky]
            phi += phi1*dky[iky]/2.*np.cos(kys[iky]*y)
            phi += phi0*dky[iky]/2.*np.cos(kys[iky]*y)
            phi0 = phi1.copy()
        return phi
