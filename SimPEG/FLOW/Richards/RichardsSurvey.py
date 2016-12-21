from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from SimPEG import Survey
from SimPEG import Utils


class BaseRichardsRx(Survey.BaseTimeRx):
    """Richards Receiver Object"""

    def __init__(self, locs, times):
        self.locs = locs
        self.times = times
        self._Ps = {}


class PressureRx(BaseRichardsRx):
    """Richards Receiver Object"""

    def __call__(self, U, m, prob):
        u = np.concatenate(U)
        return self.getP(prob.mesh, prob.timeMesh) * u

    def deriv(self, U, m, prob):
        P = self.getP(prob.mesh, prob.timeMesh)
        return P


class SaturationRx(BaseRichardsRx):
    """Richards Receiver Object"""

    def __call__(self, U, m, prob):
        prob.water_retention.model = m
        u = np.concatenate([prob.water_retention(ui) for ui in U])
        return self.getP(prob.mesh, prob.timeMesh) * u

    def deriv(self, U, m, prob):

        prob.water_retention.model = m

        P = self.getP(prob.mesh, prob.timeMesh)
        # TODO: if m is a parameter in the theta
        #       distribution, we may need to do
        #       some more chain rule here.
        dT = sp.block_diag([prob.water_retention.derivU(ui) for ui in U])
        return P*dT


class RichardsSurvey(Survey.BaseSurvey):
    """RichardsSurvey"""

    rxList = None

    def __init__(self, rxList, **kwargs):
        self.rxList = rxList
        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def nD(self):
        return np.array([rx.nD for rx in self.rxList]).sum()

    @Utils.count
    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """Create the projected data from a model.
        The field, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::
            d_\\text{pred} = P(f(m), m)

        Where P is a projection of the fields onto the data space.
        """
        if f is None:
            f = self.prob.fields(m)

        return Utils.mkvc(self(f, m))

    @Utils.requires('prob')
    def __call__(self, U, m):
        Ds = list(range(len(self.rxList)))

        for ii, rx in enumerate(self.rxList):
            Ds[ii] = rx(U, m, self.prob)

        return np.concatenate(Ds)

    @Utils.requires('prob')
    def deriv(self, U, m):
        """The Derivative with respect to the fields."""
        Ds = list(range(len(self.rxList)))

        for ii, rx in enumerate(self.rxList):
            Ds[ii] = rx.deriv(U, m, self.prob)

        return sp.vstack(Ds)
