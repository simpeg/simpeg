from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import uuid
import gc
import properties

from . import Utils
from . import Props


class BaseRx(properties.HasProperties):
    """SimPEG Receiver Object"""

    # TODO: write a validator that checks against mesh dimension in the
    # BaseSimulation
    locs = properties.Array(
        "Locations of the receivers (nRx x nDim)",
        shape=("*", "*"),
        dtype=float
    )

    knownRxTypes = properties.String(
        "Set this to a list of strings to ensure that srcType is known",
    )

    projGLoc = properties.StringChoice(
        "Projection grid location, default is CC"
        choices=["CC", "F", "E"],
        default="CC"
    )

    storeProjections = properties.Bool(
        "Store calls to getP (organized by mesh)",
        default=True
    )

    def __init__(self, locs, rxType, **kwargs):
        self.uid = str(uuid.uuid4())
        self.locs = locs
        self.rxType = rxType
        self._Ps = {}
        Utils.setKwargs(self, **kwargs)

    @property
    def rxType(self):
        """Receiver Type"""
        return getattr(self, '_rxType', None)

    @rxType.setter
    def rxType(self, value):
        known = self.knownRxTypes
        if known is not None:
            assert value in known, (
                "rxType must be in ['{0!s}']".format(("', '".join(known)))
            )
        self._rxType = value

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0]

    def getP(self, mesh, projGLoc=None):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary listed by meshes.
        """
        if mesh in self._Ps:
            return self._Ps[mesh]

        if projGLoc is None:
            projGLoc = self.projGLoc

        P = mesh.getInterpolationMat(self.locs, projGLoc)
        if self.storeProjections:
            self._Ps[mesh] = P
        return P


class BaseTimeRx(BaseRx):
    """SimPEG Receiver Object"""

    times = None   #: Times when the receivers were active.
    projTLoc = 'N'

    def __init__(self, locs, times, rxType, **kwargs):
        self.times = times
        BaseRx.__init__(self, locs, rxType, **kwargs)

    @property
    def nD(self):
        """Number of data in the receiver."""
        return self.locs.shape[0] * len(self.times)

    def getSpatialP(self, mesh):
        """
            Returns the spatial projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return mesh.getInterpolationMat(self.locs, self.projGLoc)

    def getTimeP(self, timeMesh):
        """
            Returns the time projection matrix.

            .. note::

                This is not stored in memory, but is created on demand.
        """
        return timeMesh.getInterpolationMat(self.times, self.projTLoc)

    def getP(self, mesh, timeMesh):
        """
            Returns the projection matrices as a
            list for all components collected by
            the receivers.

            .. note::

                Projection matrices are stored as a dictionary (mesh, timeMesh)
                if storeProjections is True
        """
        if (mesh, timeMesh) in self._Ps:
            return self._Ps[(mesh, timeMesh)]

        Ps = self.getSpatialP(mesh)
        Pt = self.getTimeP(timeMesh)
        P = sp.kron(Pt, Ps)

        if self.storeProjections:
            self._Ps[(mesh, timeMesh)] = P

        return P
