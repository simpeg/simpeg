from __future__ import division
import numpy as np
import scipy.sparse as sp
import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, Identity


class FieldsTDEM(SimPEG.Problem.TimeFields):
    """

    Fancy Field Storage for a TDEM survey. Only one field type is stored for
    each problem, the rest are computed. The fields obejct acts like an array
    and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[srcList,'e']
        b = f[srcList,'b']

    If accessing all sources for a given field, use the :code:`:`

    .. code-block:: python

        f = problem.fields(m)
        e = f[:,'e']
        b = f[:,'b']

    The array returned will be size (nE or nF, nSrcs :math:`\\times`
    nFrequencies)
    """

    knownFields = {}
    dtype = float

    def _eDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (self._eDeriv_u(tInd, src, v, adjoint),
                    self._eDeriv_m(tInd, src, v, adjoint))
        return (self._eDeriv_u(tInd, src, dun_dm_v) +
                self._eDeriv_m(tInd, src, v))

    def _bDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (self._bDeriv_u(tInd, src, v, adjoint),
                    self._bDeriv_m(tInd, src, v, adjoint))
        return (self._bDeriv_u(tInd, src, dun_dm_v) +
                self._bDeriv_m(tInd, src, v))


class Fields_Derivs(FieldsTDEM):
    """
        A fields object for satshing derivs
    """
    knownFields = {
                    'bDeriv': 'F',
                    'eDeriv': 'E',
                    'hDeriv': 'E',
                    'jDeriv': 'F'
                  }


class Fields3D_b(FieldsTDEM):
    """Field Storage for a TDEM survey."""
    knownFields = {'bSolution': 'F'}
    aliasFields = {
                    'b': ['bSolution', 'F', '_b'],
                    'e': ['bSolution', 'E', '_e'],
                  }

    def startup(self):
        self.MeSigmaI = self.survey.prob.MeSigmaI
        self.MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MfMui = self.survey.prob.MfMui

    def _b(self, bSolution, srcList, tInd):
        return bSolution

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return Identity()*dun_dm_v

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _e(self, bSolution, srcList, tInd):
        e = self.MeSigmaI * (self.edgeCurl.T * (self.MfMui * bSolution))
        for i, src in enumerate(srcList):
            _, s_e = src.eval(self.survey.prob, self.survey.prob.times[tInd])
            e[:, i] = e[:, i] - self.MeSigmaI * s_e
        return e

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self.MfMui.T * (self.edgeCurl * (self.MeSigmaI.T *
                                                    dun_dm_v))
        return self.MeSigmaI * (self.edgeCurl.T * (self.MfMui *
                                                   dun_dm_v))

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        _, s_e = src.eval(self.survey.prob, self.survey.prob.times[tInd])
        bSolution = self[[src], 'bSolution', tInd].flatten()

        _, s_eDeriv = src.evalDeriv(self.survey.prob.times[tInd], self,
                                    adjoint=adjoint)

        if adjoint is True:
            return (self.MeSigmaIDeriv(-s_e + self.edgeCurl.T *
                                       (self.MfMui * bSolution)).T *
                    v - s_eDeriv(self.MeSigmaI.T * v))

        return (self.MeSigmaIDeriv(-s_e + self.edgeCurl.T *
                                   (self.MfMui * bSolution)) *
                v - self.MeSigmaI * s_eDeriv(v))


class Fields3D_e(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'eSolution': 'E'}
    aliasFields = {
                    'e': ['eSolution', 'E', '_e'],
                    'b': ['eSolution', 'F', '_b'],
                    'dbdt': ['eSolution', 'F', '_dbdt'],
                  }

    def startup(self):
        self.MeSigmaI = self.survey.prob.MeSigmaI
        self.MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self.edgeCurl = self.survey.prob.mesh.edgeCurl
        self.MfMui = self.survey.prob.MfMui

    def _e(self, eSolution, srcList, tInd):
        return eSolution

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, eSolution, srcList, tInd):
        s_m = np.zeros((self.mesh.nF, 1))
        for src in srcList:
            s_m_src, _ = src.eval(self.survey.prob,
                                  self.survey.prob.times[tInd])
            s_m = s_m_src + s_m
        return s_m - self.edgeCurl * eSolution

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return -self.edgeCurl.T * dun_dm_v
        return -self.edgeCurl * dun_dm_v

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        s_mDeriv, _ = src.evalDeriv(self.survey.prob.times[tInd], self,
                                    adjoint=adjoint)
        return s_mDeriv(v)

    def _b(self, eSolution, srcList, tInd):
        """
        Integrate _db_dt using rectangles
        """
        dbdt = self._dbdt(eSolution, srcList, tInd)
        dt = self.survey.prob.timeMesh.hx
        # assume widths of "ghost cells" same on either end
        dtn = np.hstack([dt[0], 0.5*(dt[1:] + dt[:-1]), dt[-1]])
        return dtn[tInd] * dbdt
        # raise NotImplementedError

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        raise NotImplementedError

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        raise NotImplementedError


