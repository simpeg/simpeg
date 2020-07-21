from __future__ import division
import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0
from ...utils.code_utils import deprecate_class

from ...fields import TimeFields
from ...utils import mkvc, sdiag, Zero
from ..utils import omega


class FieldsTDEM(TimeFields):
    """

    Fancy Field Storage for a TDEM simulation. Only one field type is stored for
    each problem, the rest are computed. The fields obejct acts like an array
    and is indexed by

    .. code-block:: python

        f = problem.fields(m)
        e = f[source_list,'e']
        b = f[source_list,'b']

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
                self._eDeriv_m(tInd, src, v, adjoint),
            )
        return self._eDeriv_u(tInd, src, dun_dm_v) + self._eDeriv_m(tInd, src, v)

    def _bDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._bDeriv_u(tInd, src, v, adjoint),
                self._bDeriv_m(tInd, src, v, adjoint),
            )
        return self._bDeriv_u(tInd, src, dun_dm_v) + self._bDeriv_m(tInd, src, v)

    def _dbdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dbdtDeriv_u(tInd, src, v, adjoint),
                self._dbdtDeriv_m(tInd, src, v, adjoint),
            )
        return self._dbdtDeriv_u(tInd, src, dun_dm_v) + self._dbdtDeriv_m(tInd, src, v)

    def _hDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._hDeriv_u(tInd, src, v, adjoint),
                self._hDeriv_m(tInd, src, v, adjoint),
            )
        return self._hDeriv_u(tInd, src, dun_dm_v) + self._hDeriv_m(tInd, src, v)

    def _dhdtDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._dhdtDeriv_u(tInd, src, v, adjoint),
                self._dhdtDeriv_m(tInd, src, v, adjoint),
            )
        return self._dhdtDeriv_u(tInd, src, dun_dm_v) + self._dhdtDeriv_m(tInd, src, v)

    def _jDeriv(self, tInd, src, dun_dm_v, v, adjoint=False):
        if adjoint is True:
            return (
                self._jDeriv_u(tInd, src, v, adjoint),
                self._jDeriv_m(tInd, src, v, adjoint),
            )
        return self._jDeriv_u(tInd, src, dun_dm_v) + self._jDeriv_m(tInd, src, v)


class FieldsDerivativesEB(FieldsTDEM):
    """
    A fields object for satshing derivs in the EB formulation
    """

    knownFields = {
        "bDeriv": "F",
        "eDeriv": "E",
        "hDeriv": "F",
        "jDeriv": "E",
        "dbdtDeriv": "F",
        "dhdtDeriv": "F",
    }


class FieldsDerivativesHJ(FieldsTDEM):
    """
    A fields object for satshing derivs in the HJ formulation
    """

    knownFields = {
        "bDeriv": "E",
        "eDeriv": "F",
        "hDeriv": "E",
        "jDeriv": "F",
        "dbdtDeriv": "E",
        "dhdtDeriv": "E",
    }


class Fields3DMagneticFluxDensity(FieldsTDEM):
    """Field Storage for a TDEM simulation."""

    knownFields = {"bSolution": "F"}
    aliasFields = {
        "b": ["bSolution", "F", "_b"],
        "h": ["bSolution", "F", "_h"],
        "e": ["bSolution", "E", "_e"],
        "j": ["bSolution", "E", "_j"],
        "dbdt": ["bSolution", "F", "_dbdt"],
        "dhdt": ["bSolution", "F", "_dhdt"],
    }

    def startup(self):
        self._times = self.simulation.times
        self._MeSigma = self.simulation.MeSigma
        self._MeSigmaI = self.simulation.MeSigmaI
        self._MeSigmaDeriv = self.simulation.MeSigmaDeriv
        self._MeSigmaIDeriv = self.simulation.MeSigmaIDeriv
        self._edgeCurl = self.simulation.mesh.edgeCurl
        self._MfMui = self.simulation.MfMui
        self._timeMesh = self.simulation.time_mesh

    def _TLoc(self, fieldType):
        return "N"

    def _b(self, bSolution, source_list, tInd):
        return bSolution

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, bSolution, source_list, tInd):
        # self._timeMesh.faceDiv
        dbdt = -self._edgeCurl * self._e(bSolution, source_list, tInd)
        for i, src in enumerate(source_list):
            s_m = src.s_m(self.simulation, self._times[tInd])
            dbdt[:, i] = dbdt[:, i] + s_m
        return dbdt

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return -self._eDeriv_u(tInd, src, self._edgeCurl.T * dun_dm_v, adjoint)
        return -(self._edgeCurl * self._eDeriv_u(tInd, src, dun_dm_v))

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint is True:
            return -(self._eDeriv_m(tInd, src, self._edgeCurl.T * v, adjoint))
        return -(
            self._edgeCurl * self._eDeriv_m(tInd, src, v)
        )  # + src.s_mDeriv() assuming src doesn't have deriv for now

    def _e(self, bSolution, source_list, tInd):
        e = self._MeSigmaI * (self._edgeCurl.T * (self._MfMui * bSolution))
        for i, src in enumerate(source_list):
            s_e = src.s_e(self.simulation, self._times[tInd])
            e[:, i] = e[:, i] - self._MeSigmaI * s_e
        return e

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self._MfMui.T * (self._edgeCurl * (self._MeSigmaI.T * dun_dm_v))
        return self._MeSigmaI * (self._edgeCurl.T * (self._MfMui * dun_dm_v))

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        _, s_e = src.eval(self.simulation, self._times[tInd])
        bSolution = self[[src], "bSolution", tInd].flatten()

        _, s_eDeriv = src.evalDeriv(self._times[tInd], self, adjoint=adjoint)

        if adjoint is True:
            return self._MeSigmaIDeriv(
                -s_e + self._edgeCurl.T * (self._MfMui * bSolution), v, adjoint
            ) - s_eDeriv(self._MeSigmaI.T * v)

        return self._MeSigmaIDeriv(
            -s_e + self._edgeCurl.T * (self._MfMui * bSolution), v, adjoint
        ) - self._MeSigmaI * s_eDeriv(v)

    def _j(self, hSolution, source_list, tInd):
        return self.simulation.MeI * (
            self._MeSigma * self._e(hSolution, source_list, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd,
                src,
                self._MeSigma.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MeI * (
            self._MeSigma * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, "e", tInd]
        if adjoint:
            w = self.simulation.MeI.T * v
            return self._MeSigmaDeriv(e).T * w + self._eDeriv_m(
                tInd, src, self._MeSigma.T * w, adjoint=True
            )
        return self.simulation.MeI * (
            self._MeSigmaDeriv(e) * v + self._MeSigma * self._eDeriv_m(tInd, src, v)
        )

    def _h(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._MfMui * self._b(hSolution, source_list, tInd)
        )

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._bDeriv_u(
                tInd,
                src,
                self._MfMui.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (self._MfMui * self._bDeriv_u(tInd, src, dun_dm_v))

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._bDeriv_m(
                tInd, src, self._MfMui.T * (self.simulation.MfI.T * v), adjoint=True
            )
        return self.simulation.MfI * (self._MfMui * self._bDeriv_m(tInd, src, v))

    def _dhdt(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._MfMui * self._dbdt(hSolution, source_list, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd,
                src,
                self._MfMui.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._MfMui * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd, src, self._MfMui.T * (self.simulation.MfI.T * v), adjoint=True
            )
        return self.simulation.MfI * (self._MfMui * self._dbdtDeriv_m(tInd, src, v))


class Fields3DElectricField(FieldsTDEM):
    """Fancy Field Storage for a TDEM simulation."""

    knownFields = {"eSolution": "E"}
    aliasFields = {
        "e": ["eSolution", "E", "_e"],
        "j": ["eSolution", "E", "_j"],
        "b": ["eSolution", "F", "_b"],
        # 'h': ['eSolution', 'F', '_h'],
        "dbdt": ["eSolution", "F", "_dbdt"],
        "dhdt": ["eSolution", "F", "_dhdt"],
    }

    def startup(self):
        self._times = self.simulation.times
        self._MeSigma = self.simulation.MeSigma
        self._MeSigmaI = self.simulation.MeSigmaI
        self._MeSigmaDeriv = self.simulation.MeSigmaDeriv
        self._MeSigmaIDeriv = self.simulation.MeSigmaIDeriv
        self._edgeCurl = self.simulation.mesh.edgeCurl
        self._MfMui = self.simulation.MfMui

    def _TLoc(self, fieldType):
        return "N"

    def _e(self, eSolution, source_list, tInd):
        return eSolution

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, eSolution, source_list, tInd):
        s_m = np.zeros((self.mesh.nF, len(source_list)))
        for i, src in enumerate(source_list):
            s_m_src = src.s_m(self.simulation, self._times[tInd])
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
        return Zero()  # assumes source doesn't depend on model

    def _b(self, eSolution, source_list, tInd):
        """
        Integrate _db_dt using rectangles
        """
        raise NotImplementedError(
            "To obtain b-fields, please use Simulation3DMagneticFluxDensity"
        )
        # dbdt = self._dbdt(eSolution, source_list, tInd)
        # dt = self.simulation.time_mesh.hx
        # # assume widths of "ghost cells" same on either end
        # dtn = np.hstack([dt[0], 0.5*(dt[1:] + dt[:-1]), dt[-1]])
        # return dtn[tInd] * dbdt
        # # raise NotImplementedError

    def _j(self, eSolution, source_list, tInd):
        return self.simulation.MeI * (
            self._MeSigma * self._e(eSolution, source_list, tInd)
        )

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._eDeriv_u(
                tInd,
                src,
                self._MeSigma.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MeI * (
            self._MeSigma * self._eDeriv_u(tInd, src, dun_dm_v)
        )

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        e = self[src, "e", tInd]
        if adjoint:
            w = self.simulation.MeI.T * v
            return self._MeSigmaDeriv(e).T * w + self._eDeriv_m(
                tInd, src, self._MeSigma.T * w, adjoint=True
            )
        return self.simulation.MeI * (
            self._MeSigmaDeriv(e) * v + self._MeSigma * self._eDeriv_m(tInd, src, v)
        )

    def _dhdt(self, eSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._MfMui * self._dbdt(eSolution, source_list, tInd)
        )

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_u(
                tInd,
                src,
                self._MfMui.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (
            self._MfMui * self._dbdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dbdtDeriv_m(
                tInd, src, self._MfMui.T * (self.simulation.MfI.T * v)
            )
        return self.simulation.MfI * (self._MfMui * self._dbdtDeriv_m(tInd, src, v))


class Fields3DMagneticField(FieldsTDEM):
    """Fancy Field Storage for a TDEM simulation."""

    knownFields = {"hSolution": "E"}
    aliasFields = {
        "h": ["hSolution", "E", "_h"],
        "b": ["hSolution", "E", "_b"],
        "dhdt": ["hSolution", "E", "_dhdt"],
        "dbdt": ["hSolution", "E", "_dbdt"],
        "j": ["hSolution", "F", "_j"],
        "e": ["hSolution", "F", "_e"],
        "charge": ["hSolution", "CC", "_charge"],
    }

    def startup(self):
        self._times = self.simulation.times
        self._edgeCurl = self.simulation.mesh.edgeCurl
        self._MeMuI = self.simulation.MeMuI
        self._MeMu = self.simulation.MeMu
        self._MfRho = self.simulation.MfRho
        self._MfRhoDeriv = self.simulation.MfRhoDeriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return "N"
        # else:
        #     raise NotImplementedError

    def _h(self, hSolution, source_list, tInd):
        return hSolution

    def _hDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _hDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dhdt(self, hSolution, source_list, tInd):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        dhdt = -MeMuI * (C.T * (MfRho * (C * hSolution)))

        for i, src in enumerate(source_list):
            s_m, s_e = src.eval(self.simulation, self._times[tInd])
            dhdt[:, i] = MeMuI * (C.T * MfRho * s_e + s_m) + dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        if adjoint:
            return -C.T * (MfRho.T * (C * (MeMuI * dun_dm_v)))
        return -MeMuI * (C.T * (MfRho * (C * dun_dm_v)))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        C = self._edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho
        MfRhoDeriv = self._MfRhoDeriv

        hSolution = self[[src], "hSolution", tInd].flatten()
        s_e = src.s_e(self.simulation, self._times[tInd])

        if adjoint:
            return -MfRhoDeriv(C * hSolution - s_e, (C * (MeMuI * v)), adjoint)
        return -MeMuI * (C.T * (MfRhoDeriv(C * hSolution - s_e, v, adjoint)))

    def _j(self, hSolution, source_list, tInd):
        s_e = np.zeros((self.mesh.nF, len(source_list)))
        for i, src in enumerate(source_list):
            s_e_src = src.s_e(self.simulation, self._times[tInd])
            s_e[:, i] = s_e[:, i] + s_e_src

        return self._edgeCurl * hSolution - s_e

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._edgeCurl.T * dun_dm_v
        return self._edgeCurl * dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()  # assumes the source doesn't depend on the model

    def _b(self, hSolution, source_list, tInd):
        h = self._h(hSolution, source_list, tInd)
        return self.simulation.MeI * (self._MeMu * h)

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._hDeriv_u(
                tInd,
                src,
                self._MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=adjoint,
            )
        return self.simulation.MeI * (self._MeMu * self._hDeriv_u(tInd, src, dun_dm_v))

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._hDeriv_m(
                tInd, src, self._MeMu.T * (self.simulation.MeI.T * v), adjoint=adjoint
            )
        return self.simulation.MeI * (self._MeMu * self._hDeriv_m(tInd, src, v))

    def _dbdt(self, hSolution, source_list, tInd):
        dhdt = self._dhdt(hSolution, source_list, tInd)
        return self.simulation.MeI * (self._MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_u(
                tInd,
                src,
                self._MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint=adjoint,
            )
        return self.simulation.MeI * (
            self._MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src, self._MeMu.T * (self.simulation.MeI.T * v), adjoint=adjoint
            )
        return self.simulation.MeI * (self._MeMu * self._dhdtDeriv_m(tInd, src, v))

    def _e(self, hSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._MfRho * self._j(hSolution, source_list, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self._jDeriv_u(
                tInd,
                src,
                self._MfRho.T * (self.simulation.MfI.T * dun_dm_v),
                adjoint=True,
            )
        return self.simulation.MfI * (self._MfRho * self._jDeriv_u(tInd, src, dun_dm_v))

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        j = mkvc(self[src, "j", tInd])
        if adjoint is True:
            return self._MfRhoDeriv(
                j, self.simulation.MfI.T * v, adjoint
            ) + self._jDeriv_m(tInd, src, self._MfRho * v)
        return self.simulation.MfI * (
            self._MfRhoDeriv(j, v) + self._MfRho * self._jDeriv_m(tInd, src, v)
        )

    def _charge(self, hSolution, source_list, tInd):
        vol = sdiag(self.simulation.mesh.vol)
        return (
            epsilon_0
            * vol
            * (self.simulation.mesh.faceDiv * self._e(hSolution, source_list, tInd))
        )


class Fields3DCurrentDensity(FieldsTDEM):
    """Fancy Field Storage for a TDEM simulation."""

    knownFields = {"jSolution": "F"}
    aliasFields = {
        "dhdt": ["jSolution", "E", "_dhdt"],
        "dbdt": ["jSolution", "E", "_dbdt"],
        "j": ["jSolution", "F", "_j"],
        "e": ["jSolution", "F", "_e"],
        "charge": ["jSolution", "CC", "_charge"],
        "charge_density": ["jSolution", "CC", "_charge_density"],
    }

    def startup(self):
        self._times = self.simulation.times
        self._edgeCurl = self.simulation.mesh.edgeCurl
        self._MeMuI = self.simulation.MeMuI
        self._MfRho = self.simulation.MfRho
        self._MfRhoDeriv = self.simulation.MfRhoDeriv

    def _TLoc(self, fieldType):
        # if fieldType in ['h', 'j']:
        return "N"

    def _j(self, jSolution, source_list, tInd):
        return jSolution

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _h(self, jSolution, source_list, tInd):
        raise NotImplementedError(
            "Please use Simulation3DMagneticField to get h-fields"
        )

    def _dhdt(self, jSolution, source_list, tInd):
        C = self._edgeCurl
        MfRho = self._MfRho
        MeMuI = self._MeMuI

        dhdt = -MeMuI * (C.T * (MfRho * jSolution))
        for i, src in enumerate(source_list):
            s_m = src.s_m(self.simulation, self.simulation.times[tInd])
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
        jSolution = self[[src], "jSolution", tInd].flatten()
        C = self._edgeCurl
        MeMuI = self._MeMuI

        if adjoint is True:
            return -self._MfRhoDeriv(jSolution, C * (MeMuI * v), adjoint)
        return -MeMuI * (C.T * (self._MfRhoDeriv(jSolution, v)))

    def _e(self, jSolution, source_list, tInd):
        return self.simulation.MfI * (
            self._MfRho * self._j(jSolution, source_list, tInd)
        )

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return self._MfRho.T * (self.simulation.MfI.T * dun_dm_v)
        return self.simulation.MfI * (self._MfRho * dun_dm_v)

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = mkvc(self[src, "jSolution", tInd])
        if adjoint:
            return self._MfRhoDeriv(jSolution, self.simulation.MfI.T * v, adjoint)
        return self.simulation.MfI * self._MfRhoDeriv(jSolution, v)

    def _charge(self, jSolution, source_list, tInd):
        vol = sdiag(self.simulation.mesh.vol)
        return vol * self._charge_density(jSolution, source_list, tInd)

    def _charge_density(self, jSolution, source_list, tInd):
        return epsilon_0 * (
            self.simulation.mesh.faceDiv * self._e(jSolution, source_list, tInd)
        )

    def _dbdt(self, jSolution, source_list, tInd):
        dhdt = mkvc(self._dhdt(jSolution, source_list, tInd))
        return self.simulation.MeI * (self.simulation.MeMu * dhdt)

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        # dhdt = mkvc(self[src, 'dhdt', tInd])
        if adjoint:
            return self._dhdtDeriv_u(
                tInd,
                src,
                self.simulation.MeMu.T * (self.simulation.MeI.T * dun_dm_v),
                adjoint,
            )
        return self.simulation.MeI * (
            self.simulation.MeMu * self._dhdtDeriv_u(tInd, src, dun_dm_v)
        )

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint:
            return self._dhdtDeriv_m(
                tInd, src, self.simulation.MeMu.T * (self.simulation.MeI.T * v), adjoint
            )
        return self.simulation.MeI * (
            self.simulation.MeMu * self._dhdtDeriv_m(tInd, src, v)
        )


############
# Deprecated
############
@deprecate_class(removal_version="0.15.0")
class Fields_Derivs_eb(FieldsDerivativesEB):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields_Derivs_hj(FieldsDerivativesHJ):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields3D_b(Fields3DMagneticFluxDensity):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields3D_e(Fields3DElectricField):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields3D_h(Fields3DMagneticField):
    pass


@deprecate_class(removal_version="0.15.0")
class Fields3D_j(Fields3DCurrentDensity):
    pass
