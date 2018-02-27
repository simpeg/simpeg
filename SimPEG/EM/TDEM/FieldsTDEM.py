from __future__ import division
import numpy as np
import scipy.sparse as sp

from ...Fields import TimeFields
from ...Utils import Zero

__all__ = [
    'Fields3D_e', 'Fields3D_b', 'Fields3D_h', 'Fields3D_j', 'Fields_Derivs'
]


class BaseTDEMFields(TimeFields):
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

    dtype = float

    def __init__(self, **kwargs):
        super(BaseTDEMFields, self).__init__(**kwargs)

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


class Fields_Derivs(BaseTDEMFields):
    """
        A fields object for satshing derivs
    """

    def __init__(self, **kwargs):
        knownFields = {
            'bDeriv': 'F',
            'eDeriv': 'E',
            'hDeriv': 'E',
            'jDeriv': 'F',
            'dbdtDeriv': 'F',
            'dhdtDeriv': 'E'
        }

        aliasFields = {}

        knownFieldskwarg = kwargs.pop('knownFields', None)
        if knownFieldskwarg is not None:
            assert knownFieldskwarg == knownFields, (
                "knownFields should not be changed from the default"
            )

        aliasFieldskwarg = kwargs.pop('aliasFields', None)
        if knownFieldskwarg is not None:
            assert aliasFieldskwarg == aliasFields, (
                "aliasFields should not be changed from the default"
            )

        super(Fields_Derivs, self).__init__(**kwargs)

        self.knownFields = knownFields


class Fields3D_b(BaseTDEMFields):
    """Field Storage for a TDEM survey."""

    def __init__(self, **kwargs):
        knownFields = {'bSolution': 'F'}
        aliasFields = {
            'b': ['bSolution', 'F', '_b'],
            'e': ['bSolution', 'E', '_e'],
            'dbdt': ['bSolution', 'F', '_dbdt']
        }

        knownFieldskwarg = kwargs.pop('knownFields', None)
        if knownFieldskwarg is not None:
            assert knownFieldskwarg == knownFields, (
                "knownFields should not be changed from the default"
            )

        aliasFieldskwarg = kwargs.pop('aliasFields', None)
        if knownFieldskwarg is not None:
            assert aliasFieldskwarg == aliasFields, (
                "aliasFields should not be changed from the default"
            )

        super(Fields3D_b, self).__init__(**kwargs)

        self.knownFields = knownFields
        self.aliasFields = aliasFields

    def startup(self):
        self._MeSigmaI = self.simulation.MeSigmaI
        self._MeSigmaIDeriv = self.simulation.MeSigmaIDeriv
        self._MfMui = self.simulation.MfMui

    def _TLoc(self, fieldType):
        if fieldType in ['e', 'b']:
            return 'N'
        elif fieldType == 'dbdt':
            return 'N'

    def _b(self, bSolution, srcList, tInd):
        return bSolution

    def _bDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        return dun_dm_v

    def _bDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero()

    def _dbdt(self, bSolution, srcList, tInd):
        # self.simulation.time_mesh.faceDiv
        dbdt = - self.simulation.mesh.edgeCurl * self._e(bSolution, srcList, tInd)
        for i, src in enumerate(srcList):
            s_m = src.s_m(self.simulation, self.simulation.times[tInd])
            dbdt[:, i] = dbdt[:, i] + s_m
        return dbdt

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return - self._eDeriv_u(
                tInd, src, self.simulation.mesh.edgeCurl.T * dun_dm_v, adjoint
            )
        return -(self.simulation.mesh.edgeCurl * self._eDeriv_u(tInd, src, dun_dm_v))

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        if adjoint is True:
            return -(self._eDeriv_m(tInd, src, self.simulation.mesh.edgeCurl.T * v, adjoint))
        return -(self.simulation.mesh.edgeCurl * self._eDeriv_m(tInd, src, v)) #+ src.s_mDeriv() assuming src doesn't have deriv for now

    def _e(self, bSolution, srcList, tInd):
        e = self._MeSigmaI * (self.simulation.mesh.edgeCurl.T * (self._MfMui * bSolution))
        for i, src in enumerate(srcList):
            s_e = src.s_e(self.simulation, self.simulation.times[tInd])
            e[:, i] = e[:, i] - self._MeSigmaI * s_e
        return e

    def _eDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint is True:
            return (
                self._MfMui.T * (self.simulation.mesh.edgeCurl * (self._MeSigmaI.T * dun_dm_v))
            )
        return (
            self._MeSigmaI * (self.simulation.mesh.edgeCurl.T * (self._MfMui * dun_dm_v))
        )

    def _eDeriv_m(self, tInd, src, v, adjoint=False):
        _, s_e = src.eval(self.simulation, self.simulation.times[tInd])
        bSolution = self[[src], 'bSolution', tInd].flatten()

        _, s_eDeriv = src.evalDeriv(
            self.simulation.times[tInd], self, adjoint=adjoint
        )

        if adjoint is True:
            return (
                self._MeSigmaIDeriv(
                    -s_e + self.simulation.mesh.edgeCurl.T * (self._MfMui * bSolution)
                ).T * v -
                s_eDeriv(self._MeSigmaI.T * v)
            )

        return (
            self._MeSigmaIDeriv(-s_e + self.simulation.mesh.edgeCurl.T * (
                self._MfMui * bSolution)
            ) * v - self._MeSigmaI * s_eDeriv(v)
        )


class Fields3D_e(BaseTDEMFields):
    """Fancy Field Storage for a TDEM survey."""

    def __init__(self, **kwargs):

        knownFields = {'eSolution': 'E'}
        aliasFields = {
            'e': ['eSolution', 'E', '_e'],
            'b': ['eSolution', 'F', '_b'],
            'dbdt': ['eSolution', 'F', '_dbdt'],
        }

        knownFieldskwarg = kwargs.pop('knownFields', None)
        if knownFieldskwarg is not None:
            assert knownFieldskwarg == knownFields, (
                "knownFields should not be changed from the default"
            )

        aliasFieldskwarg = kwargs.pop('aliasFields', None)
        if knownFieldskwarg is not None:
            assert aliasFieldskwarg == aliasFields, (
                "aliasFields should not be changed from the default"
            )

        super(Fields3D_e, self).__init__(**kwargs)

        self.knownFields = knownFields
        self.aliasFields = aliasFields

    def startup(self):
        self._MeSigmaI = self.simulation.MeSigmaI
        self._MeSigmaIDeriv = self.simulation.MeSigmaIDeriv
        self._MfMui = self.simulation.MfMui

    def _TLoc(self, fieldType):
        if fieldType in ['e', 'b']:
            return 'N'
        elif fieldType == 'dbdt':
            return 'N'
        else:
            raise NotImplementedError

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
                self.simulation, self.simulation.times[tInd]
            )
            s_m[:, i] = s_m[:, i] + s_m_src
        return s_m - self.simulation.mesh.edgeCurl * eSolution

    def _dbdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return -self.simulation.mesh.edgeCurl.T * dun_dm_v
        return -self.simulation.mesh.edgeCurl * dun_dm_v

    def _dbdtDeriv_m(self, tInd, src, v, adjoint=False):
        # s_mDeriv = src.s_mDeriv(
        #     self.simulation.times[tInd], self, adjoint=adjoint
        # )
        return Zero()  # assumes source doesn't depend on model

    def _b(self, eSolution, srcList, tInd):
        """
        Integrate _db_dt using rectangles
        """
        raise NotImplementedError('To obtain b-fields, please use Problem3D_b')
        # dbdt = self._dbdt(eSolution, srcList, tInd)
        # dt = self.simulation.time_mesh.hx
        # # assume widths of "ghost cells" same on either end
        # dtn = np.hstack([dt[0], 0.5*(dt[1:] + dt[:-1]), dt[-1]])
        # return dtn[tInd] * dbdt
        # # raise NotImplementedError


class Fields3D_h(BaseTDEMFields):
    """Fancy Field Storage for a TDEM survey."""

    def __init__(self, **kwargs):

        knownFields = {'hSolution': 'E'}
        aliasFields = {
            'h': ['hSolution', 'E', '_h'],
            'dhdt': ['hSolution', 'E', '_dhdt'],
            'j': ['hSolution', 'F', '_j'],
        }

        knownFieldskwarg = kwargs.pop('knownFields', None)
        if knownFieldskwarg is not None:
            assert knownFieldskwarg == knownFields, (
                "knownFields should not be changed from the default"
            )

        aliasFieldskwarg = kwargs.pop('aliasFields', None)
        if knownFieldskwarg is not None:
            assert aliasFieldskwarg == aliasFields, (
                "aliasFields should not be changed from the default"
            )

        super(Fields3D_h, self).__init__(**kwargs)

        self.knownFields = knownFields
        self.aliasFields = aliasFields

    def startup(self):
        self._MeMuI = self.simulation.MeMuI
        self._MfRho = self.simulation.MfRho
        self._MfRhoDeriv = self.simulation.MfRhoDeriv

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
        C = self.simulation.mesh.edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        dhdt = - MeMuI * (C.T * (MfRho * (C * hSolution)))

        for i, src in enumerate(srcList):
            s_m, s_e = src.eval(self.simulation, self.simulation.times[tInd])
            dhdt[:, i] = MeMuI * (C.T * MfRho * s_e + s_m) +  dhdt[:, i]
        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self.simulation.mesh.edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho

        if adjoint:
            return - C.T * (MfRho.T * (C * (MeMuI * dun_dm_v)))
        return - MeMuI * (C.T * (MfRho * (C * dun_dm_v)))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        C = self.simulation.mesh.edgeCurl
        MeMuI = self._MeMuI
        MfRho = self._MfRho
        MfRhoDeriv = self._MfRhoDeriv

        hSolution = self[[src], 'hSolution', tInd].flatten()
        s_e = src.s_e(self.simulation, self.simulation.times[tInd])

        if adjoint:
            return - MfRhoDeriv(C * hSolution - s_e).T * (C * (MeMuI * v))
        return - MeMuI * (C.T * (MfRhoDeriv(C * hSolution - s_e) * v))

    def _j(self, hSolution, srcList, tInd):
        s_e = np.zeros((self.mesh.nF, len(srcList)))
        for i, src in enumerate(srcList):
            s_e_src = src.s_e(
                self.simulation, self.simulation.times[tInd]
            )
            s_e[:, i] = s_e[:, i] + s_e_src

        return self.simulation.mesh.edgeCurl * hSolution - s_e

    def _jDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        if adjoint:
            return self.simulation.mesh.edgeCurl.T * dun_dm_v
        return self.simulation.mesh.edgeCurl * dun_dm_v

    def _jDeriv_m(self, tInd, src, v, adjoint=False):
        return Zero() # assumes the source doesn't depend on the model


class Fields3D_j(BaseTDEMFields):
    """Fancy Field Storage for a TDEM survey."""

    def __init__(self, **kwargs):
        knownFields = {'jSolution': 'F'}
        aliasFields = {
            'dhdt': ['jSolution', 'E', '_dhdt'],
            'j': ['jSolution', 'F', '_j'],
        }

        knownFieldskwarg = kwargs.pop('knownFields', None)
        if knownFieldskwarg is not None:
            assert knownFieldskwarg == knownFields, (
                "knownFields should not be changed from the default"
            )

        aliasFieldskwarg = kwargs.pop('aliasFields', None)
        if knownFieldskwarg is not None:
            assert aliasFieldskwarg == aliasFields, (
                "aliasFields should not be changed from the default"
            )

        super(Fields3D_j, self).__init__(**kwargs)

        self.knownFields = knownFields
        self.aliasFields = aliasFields

    def startup(self):
        self._MeMuI = self.simulation.MeMuI
        self._MfRho = self.simulation.MfRho
        self._MfRhoDeriv = self.simulation.MfRhoDeriv

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
        C = self.simulation.mesh.edgeCurl
        MfRho = self._MfRho
        MeMuI = self._MeMuI

        dhdt = - MeMuI * (C.T * (MfRho * jSolution))
        for i, src in enumerate(srcList):
            s_m = src.s_m(self.simulation, self.simulation.times[tInd])
            dhdt[:, i] = MeMuI * s_m + dhdt[:, i]

        return dhdt

    def _dhdtDeriv_u(self, tInd, src, dun_dm_v, adjoint=False):
        C = self.simulation.mesh.edgeCurl
        MfRho = self._MfRho
        MeMuI = self._MeMuI

        if adjoint is True:
            return -MfRho.T * (C * (MeMuI.T * dun_dm_v))
        return -MeMuI * (C.T * (MfRho * dun_dm_v))

    def _dhdtDeriv_m(self, tInd, src, v, adjoint=False):
        jSolution = self[[src], 'jSolution', tInd].flatten()
        MfRhoDeriv = self._MfRhoDeriv(jSolution)
        C = self.simulation.mesh.edgeCurl
        MeMuI = self._MeMuI

        if adjoint is True:
            return -MfRhoDeriv.T * (C * (MeMuI * v))
        return -MeMuI * (C.T * (MfRhoDeriv * v))
