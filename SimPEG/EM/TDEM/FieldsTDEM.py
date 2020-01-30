from __future__ import division
import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0

import SimPEG
from SimPEG import Utils
from SimPEG.EM.Utils import omega
from SimPEG.Utils import Zero, sdiag


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

    def _GLoc(self, fieldType):
        """Grid location of the fieldType"""
        return self.aliasFields[fieldType][1]

    def _eDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._eDeriv_u(tInd, src, v, adjoint),
                self._eDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._eDeriv_u(tInd, src, dun_dm_v) +
            self._eDeriv_m(tInd, src, v)
        )

    def _bDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._bDeriv_u(tInd, src, v, adjoint),
                self._bDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._bDeriv_u(tInd, src, dun_dm_v) +
            self._bDeriv_m(tInd, src, v)
        )

    def _dbdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dbdtDeriv_u(tInd, src, v, adjoint),
                self._dbdtDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._dbdtDeriv_u(tInd, src, dun_dm_v) +
            self._dbdtDeriv_m(tInd, src, v)
        )

    def _hDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._hDeriv_u(tInd, src, v, adjoint),
                self._hDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._hDeriv_u(tInd, src, dun_dm_v) +
            self._hDeriv_m(tInd, src, v)
        )

    def _dhdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dhdtDeriv_u(tInd, src, v, adjoint),
                self._dhdtDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._dhdtDeriv_u(tInd, src, dun_dm_v) +
            self._dhdtDeriv_m(tInd, src, v)
        )

    def _jDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._jDeriv_u(tInd, src, v, adjoint),
                self._jDeriv_m(tInd, src, v, adjoint)
            )
        return (
            self._jDeriv_u(tInd, src, dun_dm_v) +
            self._jDeriv_m(tInd, src, v)
        )


class Fields_Derivs_eb(FieldsTDEM):
    """
    A fields object for satshing derivs in the EB formulatio
    """
    knownFields = {
                    'bDeriv': 'F',
                    'eDeriv': 'E',
                    'hDeriv': 'F',
                    'jDeriv': 'E',
                    'dbdtDeriv': 'F',
                    'dhdtDeriv': 'F'
                  }


class Fields_Derivs_hj(FieldsTDEM):
    """
    A fields object for satshing derivs in the HJ formulation
    """
    knownFields = {
                    'bDeriv': 'E',
                    'eDeriv': 'F',
                    'hDeriv': 'E',
                    'jDeriv': 'F',
                    'dbdtDeriv': 'E',
                    'dhdtDeriv': 'E'
                  }


class Fields3D_b(FieldsTDEM):
    """Field Storage for a TDEM survey."""
    knownFields = {'bSolution': 'F'}
    aliasFields = {
                    'b': ['bSolution', 'F', '_b'],
                    'h': ['bSolution', 'F', '_h'],
                    'e': ['bSolution', 'E', '_e'],
                    'j': ['bSolution', 'E', '_j'],
                    'dbdt': ['bSolution', 'F', '_dbdt'],
                    'dhdt': ['bSolution', 'F', '_dhdt']
                  }

    def startup(self):
        self._MeSigma = self.survey.prob.MeSigma
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MeSigmaDeriv = self.survey.prob.MeSigmaDeriv
        self._MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MfMui = self.survey.prob.MfMui
        self._timeMesh = self.survey.prob.timeMesh

    def _TLoc(self, fieldType):
        return 'N'

    def _b(self, bSolution, srcList, tInd):
        return bSolution

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, bSolution, srcList, tInd):
        # self._timeMesh.faceDiv
        dbdt = - self._edgeCurl * self._e(bSolution, srcList, tInd)
        for i, src in enumerate(srcList):
            s_m = src.s_m(self.survey.prob, self.survey.prob.times[tInd])
            dbdt[:, i] = dbdt[:, i] + s_m
        return dbdt

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return - self._eDeriv_u(
                tInd, src, self._edgeCurl.T * dun_dm_v, adjoint
            )
        return -(self._edgeCurl * self._eDeriv_u(tInd, src, dun_dm_v))

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint is True:
            return -(self._eDeriv_m(tInd, src, self._edgeCurl.T * v, adjoint))
        return -(self._edgeCurl * self._eDeriv_m(tInd, src, v)) #+ src.s_mDeriv() assuming src doesn't have deriv for now

    def _e(self, bSolution, srcList, tInd):
        e = self._MeSigmaI * (self._edgeCurl.T * (self._MfMui * bSolution))
        for i, src in enumerate(srcList):
            s_e = src.s_e(self.survey.prob, self.survey.prob.times[tInd])
            e[:, i] = e[:, i] - self._MeSigmaI * s_e
        return e

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return (
                self._MfMui.T *
                (self._edgeCurl * (self._MeSigmaI.T * dun_dm_v))
            )
        return (
            self._MeSigmaI * (self._edgeCurl.T * (self._MfMui * dun_dm_v))
        )

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        _, s_e = src.eval(self.survey.prob, self.survey.prob.times[tInd])
        bSolution = self[[src], 'bSolution', tInd].flatten()

        _, s_eDeriv = src.evalDeriv(
            self.survey.prob.times[tInd], self, adjoint=adjoint
        )

        if adjoint is True:
            return (
                self._MeSigmaIDeriv(
                    -s_e + self._edgeCurl.T * (self._MfMui * bSolution), v,
                    adjoint
                ) -
                s_eDeriv(self._MeSigmaI.T * v)
            )

        return (
            self._MeSigmaIDeriv(
                -s_e + self._edgeCurl.T * (self._MfMui * bSolution),
                v, adjoint
            ) - self._MeSigmaI * s_eDeriv(v)
        )

    def _j(self, hSolution, srcList, tInd):
        return self.survey.prob.MeI * (
            self._MeSigma * self._e(hSolution, srcList, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd, src,
                self._MeSigma.T * (self.survey.prob.MeI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MeI * (
            self._MeSigma * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, 'e', tInd]
        if adjoint:
            w = self.survey.prob.MeI.T * v
            return (
                self._MeSigmaDeriv(e).T * w +
                self._eDeriv_m(tInd, src, self._MeSigma.T * w, adjoint=True)
            )
        return self.survey.prob.MeI * (
            self._MeSigmaDeriv(e) * v +
            self._MeSigma * self._eDeriv_m(tInd, src, v)
        )

    def _h(self, hSolution, srcList, tInd):
        return self.survey.prob.MfI * (
            self._MfMui * self._b(hSolution, srcList, tInd)
        )

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._bDeriv_u(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._bDeriv_u(tInd, src, dun_dm_v)
        )

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._bDeriv_m(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._bDeriv_m(tInd, src, v)
        )

    def _dhdt(self, hSolution, srcList, tInd):
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdt(hSolution, srcList, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdtDeriv_m(tInd, src, v)
        )


class Fields3D_e(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'eSolution': 'E'}
    aliasFields = {
                    'e': ['eSolution', 'E', '_e'],
                    'j': ['eSolution', 'E', '_j'],
                    'b': ['eSolution', 'F', '_b'],
                    # 'h': ['eSolution', 'F', '_h'],
                    'dbdt': ['eSolution', 'F', '_dbdt'],
                    'dhdt': ['eSolution', 'F', '_dhdt'],
                  }

    def startup(self):
        self._MeSigma = self.survey.prob.MeSigma
        self._MeSigmaI = self.survey.prob.MeSigmaI
        self._MeSigmaDeriv = self.survey.prob.MeSigmaDeriv
        self._MeSigmaIDeriv = self.survey.prob.MeSigmaIDeriv
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._MfMui = self.survey.prob.MfMui
        self._times = self.survey.prob.times

    def _TLoc(self, fieldType):
        return 'N'

    def _e(self, eSolution, srcList, tInd):
        return eSolution

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, eSolution, srcList, tInd):
        s_m = np.zeros((self.mesh.nF, len(srcList)))
        for i, src in enumerate(srcList):
            s_m_src = src.s_m(
                self.survey.prob, self._times[tInd]
            )
            s_m[:, i] = s_m[:, i] + s_m_src
        return s_m - self._edgeCurl * eSolution

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return -self._edgeCurl.T * dun_dm_v
        return -self._edgeCurl * dun_dm_v

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        # s_mDeriv = src.s_mDeriv(
        #     self._times[tInd], self, adjoint=adjoint
        # )
        return Utils.Zero()  # assumes source doesn't depend on model

    def _b(self, eSolution, srcList, tInd):
        """
        Integrate _db_dt using rectangles
        """
        raise NotImplementedError('To obtain b-fields, please use Problem3D_b')
        # dbdt = self._dbdt(eSolution, srcList, tInd)
        # dt = self.survey.prob.timeMesh.hx
        # # assume widths of "ghost cells" same on either end
        # dtn = np.hstack([dt[0], 0.5*(dt[1:] + dt[:-1]), dt[-1]])
        # return dtn[tInd] * dbdt
        # # raise NotImplementedError

    def _j(self, eSolution, srcList, tInd):
        return self.survey.prob.MeI * (
            self._MeSigma * self._e(eSolution, srcList, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd, src,
                self._MeSigma.T * (self.survey.prob.MeI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MeI * (
            self._MeSigma * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, 'e', tInd]
        if adjoint:
            w = self.survey.prob.MeI.T * v
            return (
                self._MeSigmaDeriv(e).T * w +
                self._eDeriv_m(tInd, src, self._MeSigma.T * w, adjoint=True)
            )
        return self.survey.prob.MeI * (
            self._MeSigmaDeriv(e) * v +
            self._MeSigma * self._eDeriv_m(tInd, src, v)
        )

    def _dhdt(self, eSolution, srcList, tInd):
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdt(eSolution, srcList, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd, src, self._MfMui.T * (self.survey.prob.MfI.T * v)
            )
        return self.survey.prob.MfI * (
            self._MfMui * self._dbdtDeriv_m(tInd, src, v)
        )


class Fields3D_h(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'hSolution': 'E'}
    aliasFields = {
                    'h': ['hSolution', 'E', '_h'],
                    'b': ['hSolution', 'E', '_b'],
                    'dhdt': ['hSolution', 'E', '_dhdt'],
                    'dbdt': ['hSolution', 'E', '_dbdt'],
                    'j': ['hSolution', 'F', '_j'],
                    'e': ['hSolution', 'F', '_e'],
                    'charge': ['hSolution', 'CC', '_charge']
                  }

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._times = self.survey.prob.times
        self._MeMuI = self.survey.prob.MeMuI
        self._MeMu = self.survey.prob.MeMu
        self._MfRho = self.survey.prob.MfRho
        self._MfRhoDeriv = self.survey.prob.MfRhoDeriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return 'N'
        # else:
        #     raise NotImplementedError

    def _h(self, hSolution, srcList, tInd):
        return hSolution

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dhdt(self, hSolution, srcList, tInd):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        dhdt = - MeMuI * (C.T * (MfRho * (C * hSolution)))

        for i, src in enumerate(srcList):
            s_m, s_e = src.eval(self.survey.prob, self._times[tInd])
            dhdt[:, i] = MeMuI * (C.T * MfRho * s_e + s_m) +  dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        if adjoint:
            return - C.T * (MfRho.T * (C * (MeMuI * dun_dm_v)))
        return - MeMuI * (C.T * (MfRho * (C * dun_dm_v)))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho
        MfRhoDeriv = self._MfRhoDeriv

        hSolution = self[[src], 'hSolution', tInd].flatten()
        s_e = src.s_e(self.survey.prob, self._times[tInd])

        if adjoint:
            return - MfRhoDeriv(
                C * hSolution - s_e, (C * (MeMuI * v)), adjoint
            )
        return - MeMuI * (
            C.T * (MfRhoDeriv(C * hSolution - s_e, v, adjoint))
        )

    def _j(self, hSolution, srcList, tInd):
        s_e = np.zeros((self.mesh.nF, len(srcList)))
        for i, src in enumerate(srcList):
            s_e_src = src.s_e(
                self.survey.prob, self._times[tInd]
            )
            s_e[:, i] = s_e[:, i] + s_e_src

        return self._edgeCurl * hSolution - s_e

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._edgeCurl.T * dun_dm_v
        return self._edgeCurl * dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero() # assumes the source doesn't depend on the model

    def _b(self, hSolution, srcList, tInd):
        h = self._h(hSolution, srcList, tInd)
        return self.survey.prob.MeI * (self._MeMu * h)

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._hDeriv_u(
                tInd, src, self._MeMu.T * (self.survey.prob.MeI.T * dun_dm_v),
                adjoint=adjoint
            )
        return self.survey.prob.MeI * (self._MeMu * self._hDeriv_u(tInd, src, dun_dm_v))

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._hDeriv_m(
                tInd, src, self._MeMu.T * (self.survey.prob.MeI.T * v), adjoint=adjoint
            )
        return self.survey.prob.MeI * (self._MeMu * self._hDeriv_m(tInd, src, v))

    def _dbdt(self, hSolution, srcList, tInd):
        dhdt = self._dhdt(hSolution, srcList, tInd)
        return self.survey.prob.MeI * (self._MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_u(
                tInd, src, self._MeMu.T * (self.survey.prob.MeI.T * dun_dm_v),
                adjoint=adjoint
            )
        return self.survey.prob.MeI * (self._MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v))

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src, self._MeMu.T * (self.survey.prob.MeI.T * v), adjoint=adjoint
            )
        return self.survey.prob.MeI * (self._MeMu * self._dhdtDeriv_m(tInd, src, v))

    def _e(self, hSolution, srcList, tInd):
        return self.survey.prob.MfI * (
            self._MfRho * self._j(hSolution, srcList, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._jDeriv_u(
                tInd, src,
                self._MfRho.T * (self.survey.prob.MfI.T * dun_dm_v),
                adjoint=True
            )
        return self.survey.prob.MfI * (
            self._MfRho * self._jDeriv_u(tInd, src, dun_dm_v)
        )

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        j = Utils.mkvc(self[src, 'j', tInd])
        if adjoint is True:
            return (
                self._MfRhoDeriv(j, self.survey.prob.MfI.T * v, adjoint) +
                self._jDeriv_m(tInd, src, self._MfRho * v)
            )
        return self.survey.prob.MfI * (
            self._MfRhoDeriv(j, v) +
            self._MfRho * self._jDeriv_m(tInd, src, v)
        )

    def _charge(self, hSolution, srcList, tInd):
        vol = sdiag(self.survey.prob.mesh.vol)
        return epsilon_0*vol*(
            self.survey.prob.mesh.faceDiv*self._e(hSolution, srcList, tInd)
        )


class Fields3D_j(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'jSolution': 'F'}
    aliasFields = {
                    'dhdt': ['jSolution', 'E', '_dhdt'],
                    'dbdt': ['jSolution', 'E', '_dbdt'],
                    'j': ['jSolution', 'F', '_j'],
                    'e': ['jSolution', 'F', '_e'],
                    'charge': ['jSolution', 'CC', '_charge'],
                    'charge_density': ['jSolution', 'CC', '_charge_density'],
                  }

    def startup(self):
        self._edgeCurl = self.survey.prob.mesh.edgeCurl
        self._times = self.survey.prob.times
        self._MeMuI = self.survey.prob.MeMuI
        self._MfRho = self.survey.prob.MfRho
        self._MfRhoDeriv = self.survey.prob.MfRhoDeriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return 'N'

    def _j(self, jSolution, srcList, tInd):
        return jSolution

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _h(self, jSolution, srcList, tInd):
        raise NotImplementedError('Please use Problem3D_h to get h-fields')

    def _dhdt(self, jSolution, srcList, tInd):
        C = self._edgeCurl
        MfRho = self._MfRho
        MeMuI = self._MeMuI

        dhdt = - MeMuI * (C.T * (MfRho * jSolution))
        for i, src in enumerate(srcList):
            s_m = src.s_m(self.survey.prob, self.survey.prob.times[tInd])
            dhdt[:, i] = MeMuI * s_m + dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self._edgeCurl
        MfRho = self._MfRho
        MeMuI = self._MeMuI

        if adjoint is True:
            return -MfRho.T * (C * (MeMuI.T * dun_dm_v))
        return -MeMuI * (C.T * (MfRho * dun_dm_v))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = self[[src], 'jSolution', tInd].flatten()
        C = self._edgeCurl
        MeMuI = self._MeMuI

        if adjoint is True:
            return -self._MfRhoDeriv(jSolution, C * (MeMuI * v), adjoint)
        return -MeMuI * (C.T * (self._MfRhoDeriv(jSolution, v)))

    def _e(self, jSolution, srcList, tInd):
        return self.survey.prob.MfI * (
            self._MfRho * self._j(jSolution, srcList, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self._MfRho.T * (self.survey.prob.MfI.T * dun_dm_v)
        return self.survey.prob.MfI * (self._MfRho * dun_dm_v)

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = Utils.mkvc(self[src, 'jSolution', tInd])
        if adjoint:
            return self._MfRhoDeriv(
                jSolution, self.survey.prob.MfI.T * v, adjoint
            )
        return self.survey.prob.MfI * self._MfRhoDeriv(jSolution, v)

    def _charge(self, jSolution, srcList, tInd):
        vol = sdiag(self.survey.prob.mesh.vol)
        return vol*self._charge_density(jSolution, srcList, tInd)

    def _charge_density(self, jSolution, srcList, tInd):
        return epsilon_0*(
            self.survey.prob.mesh.faceDiv*self._e(jSolution, srcList, tInd)
        )

    def _dbdt(self, jSolution, srcList, tInd):
        dhdt = Utils.mkvc(self._dhdt(jSolution, srcList, tInd))
        return self.survey.prob.MeI * (self.survey.prob.MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        # dhdt = Utils.mkvc(self[src, 'dhdt', tInd])
        if adjoint:
            return self._dhdtDeriv_u(
                tInd, src,
                self.survey.prob.MeMu.T * (self.survey.prob.MeI.T * dun_dm_v),
                adjoint
            )
        return self.survey.prob.MeI * (
            self.survey.prob.MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src,
                self.survey.prob.MeMu.T * (self.survey.prob.MeI.T * v),
                adjoint
            )
        return self.survey.prob.MeI * (
            self.survey.prob.MeMu * self._dhdtDeriv_m(tInd, src, v)
        )

