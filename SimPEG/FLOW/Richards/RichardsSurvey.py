from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

    def __call__(self, U, prob):
        P = self.getP(prob.mesh, prob.timeMesh)
        u = np.concatenate(U)
        return P * u

    def deriv(self, U, prob, du_dm_v=None, v=None, adjoint=False):
        P = self.getP(prob.mesh, prob.timeMesh)
        if not adjoint:
            return P * du_dm_v  # + 0 for dRx_dm contribution
        assert v is not None, 'v must be provided if computing adjoint'
        return P.T * v, Utils.Zero()


class SaturationRx(BaseRichardsRx):
    """Richards Receiver Object"""

    def __call__(self, U, prob):
        # The water retention curve model should have been updated in the prob
        P = self.getP(prob.mesh, prob.timeMesh)
        usat = np.concatenate([prob.water_retention(ui) for ui in U])
        return P * usat

    def deriv(self, U, prob, du_dm_v=None, v=None, adjoint=False):
        # The water retention curve model should have been updated in the prob

        P = self.getP(prob.mesh, prob.timeMesh)
        dT_du = sp.block_diag([prob.water_retention.derivU(ui) for ui in U])

        if prob.water_retention.needs_model:
            dT_dm = sp.vstack([prob.water_retention.derivM(ui) for ui in U])
        else:
            dT_dm = Utils.Zero()

        if v is None and not adjoint:
            # this is called by the fullJ in the problem
            return P * (dT_du * du_dm_v) + P * dT_dm
        if not adjoint:
            return P * (dT_du * du_dm_v) + P * (dT_dm * v)

        # for the adjoint return both parts of the sum separately
        assert v is not None, 'v must be provided if computing adjoint'
        PTv = P.T * v
        return dT_du.T * PTv, dT_dm.T * PTv


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

        return Utils.mkvc(self(f))

    @Utils.requires('prob')
    def __call__(self, U):
        Ds = list(range(len(self.rxList)))

        for ii, rx in enumerate(self.rxList):
            Ds[ii] = rx(U, self.prob)

        return np.concatenate(Ds)

    @Utils.requires('prob')
    def deriv(self, U, du_dm_v=None, v=None):
        """The Derivative with respect to the model."""
        dd_dm = [
            rx.deriv(U, self.prob, du_dm_v=du_dm_v, v=v, adjoint=False)
            for rx in self.rxList
        ]
        return np.concatenate(dd_dm)

    @Utils.requires('prob')
    def derivAdjoint(self, U, v=None):
        """The adjoint derivative with respect to the model."""
        dd_du = list(range(len(self.rxList)))
        dd_dm = list(range(len(self.rxList)))
        cnt = 0
        for ii, rx in enumerate(self.rxList):
            dd_du[ii], dd_dm[ii] = rx.deriv(
                U, self.prob, v=v[cnt:cnt + rx.nD], adjoint=True
            )
            cnt += rx.nD
        return np.sum(dd_du, axis=0), np.sum(dd_dm, axis=0)
