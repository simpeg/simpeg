from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0

from SimPEG import utils
from ...simulation import BaseSimulation
from ..base import BasePFSimulation
from SimPEG import Solver
from SimPEG import props
# from SimPEG import Mesh
import multiprocessing
import properties
from SimPEG.utils import mkvc, matutils, sdiag
# from . import BaseMag as MAG
from .analytics import spheremodel, CongruousMagBC
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.sparse import csr_matrix as csr
from dask.delayed import Delayed
import os

class MagneticIntegralSimulation(BasePFSimulation):
    """
    magnetic simulation in integral form.

    """

    chi, chiMap, chiDeriv = props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    coordinate_system = properties.StringChoice(
        "Type of coordinate system we are regularizing in",
        choices=['cartesian', 'spherical'],
        default='cartesian'
    )

    modelType = properties.StringChoice(
        "Type of magnetization model",
        choices=['susceptibility', 'vector', 'amplitude'],
        default='susceptibility'
    )

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.modelMap = self.chiMap

    # if magType == 'H0':

    @property
    def M(self):
        """
        M: ndarray
            Magnetization matrix
        """
        if getattr(self, "_M", None) is None:

            if self.modelType == 'susceptibility':
                M = matutils.dip_azimuth2cartesian(np.ones(self.nC) * self.survey.source_field.parameters[1],
                                          np.ones(self.nC) * self.survey.source_field.parameters[2])

                Mx = sdiag(M[:, 0] * self.survey.source_field.parameters[0])
                My = sdiag(M[:, 1] * self.survey.source_field.parameters[0])
                Mz = sdiag(M[:, 2] * self.survey.source_field.parameters[0])

                self._M = sp.vstack((Mx, My, Mz))

            else:

                self._M = sp.identity(3*self.nC) * self.survey.source_field.parameters[0]

        return self._M

    def fields(self, m):

        if self.coordinate_system == 'cartesian':
            m = self.chiMap*(m)
        else:
            m = self.chiMap*(matutils.spherical2cartesian(m.reshape((int(len(m)/3), 3), order='F')))

        if self.store_sensitivities == 'forward_only':
            self.model = m
            # Compute the linear operation without forming the full dense G
            return mkvc(self.linear_operator())

        # TO-DO: Delay the fields all the way to the objective function
        if getattr(self, '_Mxyz', None) is not None:

            vec = dask.delayed(csr.dot)(self.M, m)
            M = da.from_delayed(vec, dtype=float, shape=[m.shape[0]])
            fields = da.dot(self.G, M).compute()

        else:

            fields = da.dot(self.G, m.astype(np.float32)).compute()

        if self.modelType == 'amplitude':

            fields = self.calcAmpData(fields)

        return fields

    def calcAmpData(self, Bxyz):
        """
            Compute amplitude of the field
        """

        amplitude = da.sum(
            Bxyz.reshape((3, self.nD), order='F')**2., axis=0
        )**0.5

        return amplitude

    @property
    def G(self):

        if getattr(self, '_G', None) is None:

            self._G = self.linear_operator()

        return self._G

    @property
    def nD(self):
        """
            Number of data
        """
        self._nD = self.survey.receiver_locations.shape[0]

        return self._nD

    @property
    def ProjTMI(self):

        if getattr(self, '_ProjTMI', None) is None:

            # Convert Bdecination from north to cartesian
            self._ProjTMI = matutils.dip_azimuth2cartesian(
                self.survey.source_field.parameters[1],
                self.survey.source_field.parameters[2]
            )

        return self._ProjTMI

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        dmudm = self.chiMap.deriv(m)
        self._dSdm = None
        self._dfdm = None
        self.model = m
        if (
            getattr(self, "gtgdiag", None) is None and
            self.modelType != 'amplitude'
            ):

            if W is None:
                w = np.ones(self.G.shape[1])
            else:
                w = W.diagonal()

            self.gtgdiag = np.array(da.sum(da.power(self.G, 2), axis=0))

        if self.coordinate_system == 'cartesian':
            if self.modelType == 'amplitude':
                return np.sum((W * self.dfdm * sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0)
            else:
                return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * dmudm).power(2.), axis=0))

        else:  # spherical
            if self.modelType == 'amplitude':
                return mkvc(np.sum(((W * self.dfdm) * sdiag(mkvc(self.gtgdiag)**0.5) * (self.dSdm * dmudm)).power(2.), axis=0))
            else:

                return mkvc(np.sum((sdiag(mkvc(self.gtgdiag)**0.5) * self.dSdm * dmudm).power(2), axis=0))

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:  # spherical
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == 'amplitude':
            return self.dfdm * da.dot(self.G, dmudm)
        else:

            prod = dask.delayed(
                csr.dot)(
                    self.G, dmudm
                )
            return da.from_delayed(
                prod, dtype=float,
                shape=(self.G.shape[0], dmudm.shape[1])
            )

    def Jvec(self, m, v, f=None):

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = self.dSdm * self.chiMap.deriv(m)



        if getattr(self, '_Mxyz', None) is not None:

            M_dmudm_v = da.from_array(self.M*(dmudm*v), chunks=self.G.chunks[1])

            # TO-DO: Delay the fields all the way to the objective function
            Jvec = da.dot(self.G, M_dmudm_v.astype(np.float32))

        else:

            if isinstance(dmudm, Delayed):
                dmudm_v = da.from_array(dmudm*v, chunks=self.G.chunks[1])
            else:
                dmudm_v = dmudm * v

            Jvec = da.dot(self.G, dmudm_v.astype(np.float32))

        if self.modelType == 'amplitude':
            dfdm_Jvec = dask.delayed(csr.dot)(self.dfdm, Jvec)

            return da.from_delayed(dfdm_Jvec, dtype=float, shape=[self.dfdm.shape[0]])
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):

        if self.coordinate_system == 'cartesian':
            dmudm = self.chiMap.deriv(m)
        else:
            dmudm = self.dSdm * self.chiMap.deriv(m)

        if self.modelType == 'amplitude':

            dfdm_v = dask.delayed(csr.dot)(v, self.dfdm)

            vec = da.from_delayed(dfdm_v, dtype=float, shape=[self.dfdm.shape[0]])

            if getattr(self, '_Mxyz', None) is not None:

                jtvec = da.dot(vec.astype(np.float32), self.G)

                Jtvec = dask.delayed(csr.dot)(jtvec, self.M)

            else:
                Jtvec = da.dot(vec.astype(np.float32), self.G)

        else:

            Jtvec = da.dot(v.astype(np.float32), self.G)

        dmudm_v = dask.delayed(csr.dot)(Jtvec, dmudm)

        return da.from_delayed(dmudm_v, dtype=float, shape=[dmudm.shape[1]]).compute()

    @property
    def dSdm(self):

        if getattr(self, '_dSdm', None) is None:

            if self.model is None:
                raise Exception('Requires a chi')

            nC = int(len(self.model)/3)

            m_xyz = self.chiMap * matutils.spherical2cartesian(self.model.reshape((nC, 3), order='F'))

            nC = int(m_xyz.shape[0]/3.)
            m_atp = matutils.cartesian2spherical(m_xyz.reshape((nC, 3), order='F'))

            a = m_atp[:nC]
            t = m_atp[nC:2*nC]
            p = m_atp[2*nC:]

            Sx = sp.hstack([sp.diags(np.cos(t)*np.cos(p), 0),
                            sp.diags(-a*np.sin(t)*np.cos(p), 0),
                            sp.diags(-a*np.cos(t)*np.sin(p), 0)])

            Sy = sp.hstack([sp.diags(np.cos(t)*np.sin(p), 0),
                            sp.diags(-a*np.sin(t)*np.sin(p), 0),
                            sp.diags(a*np.cos(t)*np.cos(p), 0)])

            Sz = sp.hstack([sp.diags(np.sin(t), 0),
                            sp.diags(a*np.cos(t), 0),
                            csr((nC, nC))])

            self._dSdm = sp.vstack([Sx, Sy, Sz])

        return self._dSdm

    @property
    def dfdm(self):

        if self.model is None:
            self.model = np.zeros(self.G.shape[1])

        if getattr(self, '_dfdm', None) is None:

            Bxyz = self.Bxyz_a(self.chiMap * self.model)

            ii = np.kron(np.asarray(range(self.survey.nD), dtype='int'), np.ones(3))
            jj = np.asarray(range(3*self.survey.nD), dtype='int')

            self._dfdm = csr((mkvc(Bxyz), (ii, jj)), shape=(self.survey.nD, 3*self.survey.nD))

        return self._dfdm

    def Bxyz_a(self, m):
        """
            Return the normalized B fields
        """

        # Get field data
        if self.coordinate_system == 'spherical':
            m = matutils.spherical2cartesian(m)

        if getattr(self, '_Mxyz', None) is not None:
            Bxyz = da.dot(self.G, (self.M*m).astype(np.float32))
        else:
            Bxyz = da.dot(self.G, m.astype(np.float32))

        amp = self.calcAmpData(Bxyz.astype(np.float64))
        Bamp = sp.spdiags(1./amp, 0, self.nD, self.nD)

        return (Bxyz.reshape((3, self.nD), order='F')*Bamp)

    def evaluate_integral(self, receiver_location, components):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            receiver_location:  [obsx, obsy, obsz] nC x 3 Array

            components: list[str]
                List of gravity components chosen from:
                'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]
        """

        eps = 1e-8  # add a small value to the locations to avoid /0

        rows = {component: np.zeros(3*self.Xn.shape[0]) for component in components}

        # number of cells in mesh
        nC = self.Xn.shape[0]

        # comp. pos. differences for tne, bsw nodes
        dz2 = self.Zn[:, 1] - receiver_location[2] + eps
        dz1 = self.Zn[:, 0] - receiver_location[2] + eps

        dy2 = self.Yn[:, 1] - receiver_location[1] + eps
        dy1 = self.Yn[:, 0] - receiver_location[1] + eps

        dx2 = self.Xn[:, 1] - receiver_location[0] + eps
        dx1 = self.Xn[:, 0] - receiver_location[0] + eps

        # comp. squared diff
        dx2dx2 = dx2**2.
        dx1dx1 = dx1**2.

        dy2dy2 = dy2**2.
        dy1dy1 = dy1**2.

        dz2dz2 = dz2**2.
        dz1dz1 = dz1**2.

        # 2D radius component squared of corner nodes
        R1 = (dy2dy2 + dx2dx2)
        R2 = (dy2dy2 + dx1dx1)
        R3 = (dy1dy1 + dx2dx2)
        R4 = (dy1dy1 + dx1dx1)

        # radius to each cell node
        r1 = np.sqrt(dz2dz2 + R2) + eps
        r2 = np.sqrt(dz2dz2 + R1) + eps
        r3 = np.sqrt(dz1dz1 + R1) + eps
        r4 = np.sqrt(dz1dz1 + R2) + eps
        r5 = np.sqrt(dz2dz2 + R3) + eps
        r6 = np.sqrt(dz2dz2 + R4) + eps
        r7 = np.sqrt(dz1dz1 + R4) + eps
        r8 = np.sqrt(dz1dz1 + R3) + eps

        # compactify argument calculations
        arg1_ = dx1 + dy2 + r1
        arg1 = dy2 + dz2 + r1
        arg2 = dx1 + dz2 + r1
        arg3 = dx1 + r1
        arg4 = dy2 + r1
        arg5 = dz2 + r1

        arg6_ = dx2 + dy2 + r2
        arg6 = dy2 + dz2 + r2
        arg7 = dx2 + dz2 + r2
        arg8 = dx2 + r2
        arg9 = dy2 + r2
        arg10 = dz2 + r2

        arg11_ = dx2 + dy2 + r3
        arg11 = dy2 + dz1 + r3
        arg12 = dx2 + dz1 + r3
        arg13 = dx2 + r3
        arg14 = dy2 + r3
        arg15 = dz1 + r3

        arg16_ = dx1 + dy2 + r4
        arg16 = dy2 + dz1 + r4
        arg17 = dx1 + dz1 + r4
        arg18 = dx1 + r4
        arg19 = dy2 + r4
        arg20 = dz1 + r4

        arg21_ = dx2 + dy1 + r5
        arg21 = dy1 + dz2 + r5
        arg22 = dx2 + dz2 + r5
        arg23 = dx2 + r5
        arg24 = dy1 + r5
        arg25 = dz2 + r5

        arg26_ = dx1 + dy1 + r6
        arg26 = dy1 + dz2 + r6
        arg27 = dx1 + dz2 + r6
        arg28 = dx1 + r6
        arg29 = dy1 + r6
        arg30 = dz2 + r6

        arg31_ = dx1 + dy1 + r7
        arg31 = dy1 + dz1 + r7
        arg32 = dx1 + dz1 + r7
        arg33 = dx1 + r7
        arg34 = dy1 + r7
        arg35 = dz1 + r7

        arg36_ = dx2 + dy1 + r8
        arg36 = dy1 + dz1 + r8
        arg37 = dx2 + dz1 + r8
        arg38 = dx2 + r8
        arg39 = dy1 + r8
        arg40 = dz1 + r8

        if ("bxx" in components) or ("bzz" in components):
            rows["bxx"] = np.zeros((1, 3 * nC))

            rows["bxx"][0, 0:nC] = (
                2 * (
                    (
                        (dx1**2 - r1 * arg1) /
                        (r1 * arg1**2 + dx1**2 * r1 + eps)
                    ) -
                    (
                        (dx2**2 - r2 * arg6) /
                        (r2 * arg6**2 + dx2**2 * r2 + eps)
                    ) +
                    (
                        (dx2**2 - r3 * arg11) /
                        (r3 * arg11**2 + dx2**2 * r3 + eps)
                    ) -
                    (
                        (dx1**2 - r4 * arg16) /
                        (r4 * arg16**2 + dx1**2 * r4 + eps)
                    ) +
                    (
                        (dx2**2 - r5 * arg21) /
                        (r5 * arg21**2 + dx2**2 * r5 + eps)
                    ) -
                    (
                        (dx1**2 - r6 * arg26) /
                        (r6 * arg26**2 + dx1**2 * r6 + eps)
                    ) +
                    (
                        (dx1**2 - r7 * arg31) /
                        (r7 * arg31**2 + dx1**2 * r7 + eps)
                    ) -
                    (
                        (dx2**2 - r8 * arg36) /
                        (r8 * arg36**2 + dx2**2 * r8 + eps)
                    )
                )
            )

            rows["bxx"][0, nC:2*nC] = (
                dx2 / (r5 * arg25 + eps) - dx2 / (r2 * arg10 + eps) +
                dx2 / (r3 * arg15 + eps) - dx2 / (r8 * arg40 + eps) +
                dx1 / (r1 * arg5 + eps) - dx1 / (r6 * arg30 + eps) +
                dx1 / (r7 * arg35 + eps) - dx1 / (r4 * arg20 + eps)
            )

            rows["bxx"][0, 2*nC:] = (
                dx1 / (r1 * arg4 + eps) - dx2 / (r2 * arg9 + eps) +
                dx2 / (r3 * arg14 + eps) - dx1 / (r4 * arg19 + eps) +
                dx2 / (r5 * arg24 + eps) - dx1 / (r6 * arg29 + eps) +
                dx1 / (r7 * arg34 + eps) - dx2 / (r8 * arg39 + eps)
            )

            rows["bxx"] /= (4 * np.pi)
            rows["bxx"] *= self.M

        if ("byy" in components) or ("bzz" in components):

            rows["byy"] = np.zeros((1, 3 * nC))

            rows["byy"][0, 0:nC] = (dy2 / (r3 * arg15 + eps) - dy2 / (r2 * arg10 + eps) +
                        dy1 / (r5 * arg25 + eps) - dy1 / (r8 * arg40 + eps) +
                        dy2 / (r1 * arg5 + eps) - dy2 / (r4 * arg20 + eps) +
                        dy1 / (r7 * arg35 + eps) - dy1 / (r6 * arg30 + eps))
            rows["byy"][0, nC:2*nC] = (2 * (((dy2**2 - r1 * arg2) / (r1 * arg2**2 + dy2**2 * r1 + eps)) -
                       ((dy2**2 - r2 * arg7) / (r2 * arg7**2 + dy2**2 * r2 + eps)) +
                       ((dy2**2 - r3 * arg12) / (r3 * arg12**2 + dy2**2 * r3 + eps)) -
                       ((dy2**2 - r4 * arg17) / (r4 * arg17**2 + dy2**2 * r4 + eps)) +
                       ((dy1**2 - r5 * arg22) / (r5 * arg22**2 + dy1**2 * r5 + eps)) -
                       ((dy1**2 - r6 * arg27) / (r6 * arg27**2 + dy1**2 * r6 + eps)) +
                       ((dy1**2 - r7 * arg32) / (r7 * arg32**2 + dy1**2 * r7 + eps)) -
                       ((dy1**2 - r8 * arg37) / (r8 * arg37**2 + dy1**2 * r8 + eps))))
            rows["byy"][0, 2*nC:] = (dy2 / (r1 * arg3 + eps) - dy2 / (r2 * arg8 + eps) +
                         dy2 / (r3 * arg13 + eps) - dy2 / (r4 * arg18 + eps) +
                         dy1 / (r5 * arg23 + eps) - dy1 / (r6 * arg28 + eps) +
                         dy1 / (r7 * arg33 + eps) - dy1 / (r8 * arg38 + eps))

            rows["byy"] /= (4 * np.pi)
            rows["byy"] *= self.M

        if "bzz" in components:

            rows["bzz"] = -rows["bxx"] - rows["byy"]

        if "bxy" in components:
            rows["bxy"] = np.zeros((1, 3 * nC))

            rows["bxy"][0, 0:nC] = (2 * (((dx1 * arg4) / (r1 * arg1**2 + (dx1**2) * r1 + eps)) -
                        ((dx2 * arg9) / (r2 * arg6**2 + (dx2**2) * r2 + eps)) +
                        ((dx2 * arg14) / (r3 * arg11**2 + (dx2**2) * r3 + eps)) -
                        ((dx1 * arg19) / (r4 * arg16**2 + (dx1**2) * r4 + eps)) +
                        ((dx2 * arg24) / (r5 * arg21**2 + (dx2**2) * r5 + eps)) -
                        ((dx1 * arg29) / (r6 * arg26**2 + (dx1**2) * r6 + eps)) +
                        ((dx1 * arg34) / (r7 * arg31**2 + (dx1**2) * r7 + eps)) -
                        ((dx2 * arg39) / (r8 * arg36**2 + (dx2**2) * r8 + eps))))
            rows["bxy"][0, nC:2*nC] = (dy2 / (r1 * arg5 + eps) - dy2 / (r2 * arg10 + eps) +
                           dy2 / (r3 * arg15 + eps) - dy2 / (r4 * arg20 + eps) +
                           dy1 / (r5 * arg25 + eps) - dy1 / (r6 * arg30 + eps) +
                           dy1 / (r7 * arg35 + eps) - dy1 / (r8 * arg40 + eps))
            rows["bxy"][0, 2*nC:] = (1 / r1 - 1 / r2 +
                         1 / r3 - 1 / r4 +
                         1 / r5 - 1 / r6 +
                         1 / r7 - 1 / r8)

            rows["bxy"] /= (4 * np.pi)

            rows["bxy"] *= self.M

        if "bxz" in components:
            rows["bxz"] = np.zeros((1, 3 * nC))

            rows["bxz"][0, 0:nC] =(2 * (((dx1 * arg5) / (r1 * (arg1**2) + (dx1**2) * r1 + eps)) -
                        ((dx2 * arg10) / (r2 * (arg6**2) + (dx2**2) * r2 + eps)) +
                        ((dx2 * arg15) / (r3 * (arg11**2) + (dx2**2) * r3 + eps)) -
                        ((dx1 * arg20) / (r4 * (arg16**2) + (dx1**2) * r4 + eps)) +
                        ((dx2 * arg25) / (r5 * (arg21**2) + (dx2**2) * r5 + eps)) -
                        ((dx1 * arg30) / (r6 * (arg26**2) + (dx1**2) * r6 + eps)) +
                        ((dx1 * arg35) / (r7 * (arg31**2) + (dx1**2) * r7 + eps)) -
                        ((dx2 * arg40) / (r8 * (arg36**2) + (dx2**2) * r8 + eps))))
            rows["bxz"][0, nC:2*nC] = (1 / r1 - 1 / r2 +
                           1 / r3 - 1 / r4 +
                           1 / r5 - 1 / r6 +
                           1 / r7 - 1 / r8)
            rows["bxz"][0, 2*nC:] = (dz2 / (r1 * arg4 + eps) - dz2 / (r2 * arg9 + eps) +
                         dz1 / (r3 * arg14 + eps) - dz1 / (r4 * arg19 + eps) +
                         dz2 / (r5 * arg24 + eps) - dz2 / (r6 * arg29 + eps) +
                         dz1 / (r7 * arg34 + eps) - dz1 / (r8 * arg39 + eps))

            rows["bxz"] /= (4 * np.pi)

            rows["bxz"] *= self.M

        if "byz" in components:
            rows["byz"] = np.zeros((1, 3 * nC))

            rows["byz"][0, 0:nC] = (1 / r3 - 1 / r2 +
                        1 / r5 - 1 / r8 +
                        1 / r1 - 1 / r4 +
                        1 / r7 - 1 / r6)
            rows["byz"][0, nC:2*nC] = (2 * ((((dy2 * arg5) / (r1 * (arg2**2) + (dy2**2) * r1 + eps))) -
                    (((dy2 * arg10) / (r2 * (arg7**2) + (dy2**2) * r2 + eps))) +
                    (((dy2 * arg15) / (r3 * (arg12**2) + (dy2**2) * r3 + eps))) -
                    (((dy2 * arg20) / (r4 * (arg17**2) + (dy2**2) * r4 + eps))) +
                    (((dy1 * arg25) / (r5 * (arg22**2) + (dy1**2) * r5 + eps))) -
                    (((dy1 * arg30) / (r6 * (arg27**2) + (dy1**2) * r6 + eps))) +
                    (((dy1 * arg35) / (r7 * (arg32**2) + (dy1**2) * r7 + eps))) -
                    (((dy1 * arg40) / (r8 * (arg37**2) + (dy1**2) * r8 + eps)))))
            rows["byz"][0, 2*nC:] = (dz2 / (r1 * arg3  + eps) - dz2 / (r2 * arg8 + eps) +
                     dz1 / (r3 * arg13 + eps) - dz1 / (r4 * arg18 + eps) +
                     dz2 / (r5 * arg23 + eps) - dz2 / (r6 * arg28 + eps) +
                     dz1 / (r7 * arg33 + eps) - dz1 / (r8 * arg38 + eps))

            rows["byz"] /= (4 * np.pi)

            rows["byz"] *= self.M

        if ("bx" in components) or ("tmi" in components):
            rows["bx"] = np.zeros((1, 3 * nC))

            rows["bx"][0, 0:nC] = ((-2 * np.arctan2(dx1, arg1 + eps)) - (-2 * np.arctan2(dx2, arg6 + eps)) +
                       (-2 * np.arctan2(dx2, arg11 + eps)) - (-2 * np.arctan2(dx1, arg16 + eps)) +
                       (-2 * np.arctan2(dx2, arg21 + eps)) - (-2 * np.arctan2(dx1, arg26 + eps)) +
                       (-2 * np.arctan2(dx1, arg31 + eps)) - (-2 * np.arctan2(dx2, arg36 + eps)))
            rows["bx"][0, nC:2*nC] = (np.log(arg5) - np.log(arg10) +
                          np.log(arg15) - np.log(arg20) +
                          np.log(arg25) - np.log(arg30) +
                          np.log(arg35) - np.log(arg40))
            rows["bx"][0, 2*nC:] = ((np.log(arg4) - np.log(arg9)) +
                        (np.log(arg14) - np.log(arg19)) +
                        (np.log(arg24) - np.log(arg29)) +
                        (np.log(arg34) - np.log(arg39)))
            rows["bx"] /= (4 * np.pi)

            rows["bx"] *= self.M

        if ("by" in components) or ("tmi" in components):
            rows["by"] = np.zeros((1, 3 * nC))

            rows["by"][0, 0:nC] = (np.log(arg5) - np.log(arg10) +
                       np.log(arg15) - np.log(arg20) +
                       np.log(arg25) - np.log(arg30) +
                       np.log(arg35) - np.log(arg40))
            rows["by"][0, nC:2*nC] = ((-2 * np.arctan2(dy2, arg2 + eps)) - (-2 * np.arctan2(dy2, arg7 + eps)) +
                              (-2 * np.arctan2(dy2, arg12 + eps)) - (-2 * np.arctan2(dy2, arg17 + eps)) +
                              (-2 * np.arctan2(dy1, arg22 + eps)) - (-2 * np.arctan2(dy1, arg27 + eps)) +
                              (-2 * np.arctan2(dy1, arg32 + eps)) - (-2 * np.arctan2(dy1, arg37 + eps)))
            rows["by"][0, 2*nC:] = ((np.log(arg3) - np.log(arg8)) +
                            (np.log(arg13) - np.log(arg18)) +
                            (np.log(arg23) - np.log(arg28)) +
                            (np.log(arg33) - np.log(arg38)))

            rows["by"] /= (-4 * np.pi)

            rows["by"] *= self.M

        if ("bz" in components) or ("tmi" in components):
            rows["bz"] = np.zeros((1, 3 * nC))

            rows["bz"][0, 0:nC] = (np.log(arg4) - np.log(arg9) +
                       np.log(arg14) - np.log(arg19) +
                       np.log(arg24) - np.log(arg29) +
                       np.log(arg34) - np.log(arg39))
            rows["bz"][0, nC:2*nC] = ((np.log(arg3) - np.log(arg8)) +
                              (np.log(arg13) - np.log(arg18)) +
                              (np.log(arg23) - np.log(arg28)) +
                              (np.log(arg33) - np.log(arg38)))
            rows["bz"][0, 2*nC:] = ((-2 * np.arctan2(dz2, arg1_ + eps)) - (-2 * np.arctan2(dz2, arg6_ + eps)) +
                            (-2 * np.arctan2(dz1, arg11_ + eps)) - (-2 * np.arctan2(dz1, arg16_ + eps)) +
                            (-2 * np.arctan2(dz2, arg21_ + eps)) - (-2 * np.arctan2(dz2, arg26_ + eps)) +
                            (-2 * np.arctan2(dz1, arg31_ + eps)) - (-2 * np.arctan2(dz1, arg36_ + eps)))
            rows["bz"] /= (-4 * np.pi)

            rows["bz"] *= self.M

        if "tmi" in components:

            rows["tmi"] = np.dot(self.ProjTMI, np.r_[rows["bx"], rows["by"], rows["bz"]])

        if self.store_sensitivities == "forward_only":
            return np.dot(
                np.vstack([rows[component] for component in components]),
                self.model
            )
        else:

            return np.vstack([rows[component] for component in components])


class DifferentialEquationSimulation(BaseSimulation):
    """
        Secondary field approach using differential equations!
    """

    # surveyPair = MAG.BaseMagSurvey
    # modelPair = MAG.BaseMagMap

    mu, muMap, muDeriv = props.Invertible(
        "Magnetic Permeability (H/m)",
        default=mu_0
    )

    mui, muiMap, muiDeriv = props.Invertible(
        "Inverse Magnetic Permeability (m/H)"
    )

    props.Reciprocal(mu, mui)

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        Pbc, Pin, self._Pout = \
            self.mesh.getBCProjWF('neumann', discretization='CC')

        Dface = self.mesh.faceDiv
        Mc = sdiag(self.mesh.vol)
        self._Div = Mc * Dface * Pin.T * Pin

    @property
    def MfMuI(self): return self._MfMuI

    @property
    def MfMui(self): return self._MfMui

    @property
    def MfMu0(self): return self._MfMu0

    def makeMassMatrices(self, m):
        mu = self.muMap * m
        self._MfMui = self.mesh.getFaceInnerProduct(1. / mu) / self.mesh.dim
        # self._MfMui = self.mesh.getFaceInnerProduct(1./mu)
        # TODO: this will break if tensor mu
        self._MfMuI = sdiag(1. / self._MfMui.diagonal())
        self._MfMu0 = self.mesh.getFaceInnerProduct(1. / mu_0) / self.mesh.dim
        # self._MfMu0 = self.mesh.getFaceInnerProduct(1/mu_0)

    @utils.requires('survey')
    def getB0(self):
        b0 = self.survey.B0
        B0 = np.r_[
            b0[0] * np.ones(self.mesh.nFx),
            b0[1] * np.ones(self.mesh.nFy),
            b0[2] * np.ones(self.mesh.nFz)
        ]
        return B0

    def getRHS(self, m):
        """

        .. math ::

            \mathbf{rhs} = \Div(\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0 - \Div\mathbf{B}_0+\diag(v)\mathbf{D} \mathbf{P}_{out}^T \mathbf{B}_{sBC}

        """
        B0 = self.getB0()
        Dface = self.mesh.faceDiv
        # Mc = sdiag(self.mesh.vol)

        mu = self.muMap * m
        chi = mu / mu_0 - 1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.B0, chi)
        self.Bbc = Bbc
        self.Bbc_const = Bbc_const
        # return self._Div*self.MfMuI*self.MfMu0*B0 - self._Div*B0 +
        # Mc*Dface*self._Pout.T*Bbc
        return self._Div * self.MfMuI * self.MfMu0 * B0 - self._Div * B0

    def getA(self, m):
        """
        GetA creates and returns the A matrix for the Magnetics problem

        The A matrix has the form:

        .. math ::

            \mathbf{A} =  \Div(\MfMui)^{-1}\Div^{T}

        """
        return self._Div * self.MfMuI * self._Div.T

    def fields(self, m):
        """
            Return magnetic potential (u) and flux (B)
            u: defined on the cell center [nC x 1]
            B: defined on the cell center [nG x 1]

            After we compute u, then we update B.

            .. math ::

                \mathbf{B}_s = (\MfMui)^{-1}\mathbf{M}^f_{\mu_0^{-1}}\mathbf{B}_0-\mathbf{B}_0 -(\MfMui)^{-1}\Div^T \mathbf{u}

        """
        self.makeMassMatrices(m)
        A = self.getA(m)
        rhs = self.getRHS(m)
        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / A.diagonal())
        )
        u, info = sp.linalg.bicgstab(A, rhs, tol=1e-6, maxiter=1000, M=m1)
        B0 = self.getB0()
        B = self.MfMuI * self.MfMu0 * B0 - B0 - self.MfMuI * self._Div.T * u

        return {'B': B, 'u': u}

    @utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Computing Jacobian multiplied by vector

            By setting our problem as

            .. math ::

                \mathbf{C}(\mathbf{m}, \mathbf{u}) = \mathbf{A}\mathbf{u} - \mathbf{rhs} = 0

            And taking derivative w.r.t m

            .. math ::

                \\nabla \mathbf{C}(\mathbf{m}, \mathbf{u}) = \\nabla_m \mathbf{C}(\mathbf{m}) \delta \mathbf{m} +
                                                             \\nabla_u \mathbf{C}(\mathbf{u}) \delta \mathbf{u} = 0

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} = - [\\nabla_u \mathbf{C}(\mathbf{u})]^{-1}\\nabla_m \mathbf{C}(\mathbf{m})

            With some linear algebra we can have

            .. math ::

                \\nabla_u \mathbf{C}(\mathbf{u}) = \mathbf{A}

                \\nabla_m \mathbf{C}(\mathbf{m}) =
                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}}

            .. math ::

                \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u} =
                \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[\Div \diag (\Div^T \mathbf{u}) \dMfMuI \\right]

                \dMfMuI = \diag(\MfMui)^{-1}_{vec} \mathbf{Av}_{F2CC}^T\diag(\mathbf{v})\diag(\\frac{1}{\mu^2})

                \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} =  \\frac{\partial \mathbf{\mu}}{\partial \mathbf{m}} \left[
                \Div \diag(\M^f_{\mu_{0}^{-1}}\mathbf{B}_0) \dMfMuI \\right] - \diag(\mathbf{v})\mathbf{D} \mathbf{P}_{out}^T\\frac{\partial B_{sBC}}{\partial \mathbf{m}}

            In the end,

            .. math ::

                \\frac{\delta \mathbf{u}}{\delta \mathbf{m}} =
                - [ \mathbf{A} ]^{-1}\left[ \\frac{\partial \mathbf{A}}{\partial \mathbf{m}}(\mathbf{m})\mathbf{u}
                - \\frac{\partial \mathbf{rhs}(\mathbf{m})}{\partial \mathbf{m}} \\right]

            A little tricky point here is we are not interested in potential (u), but interested in magnetic flux (B).
            Thus, we need sensitivity for B. Now we take derivative of B w.r.t m and have

            .. math ::

                \\frac{\delta \mathbf{B}} {\delta \mathbf{m}} = \\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
                \left[
                \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
                 -  \diag (\Div^T\mathbf{u})\dMfMuI
                \\right ]

                 -  (\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}}

            Finally we evaluate the above, but we should remember that

            .. note ::

                We only want to evalute

                .. math ::

                    \mathbf{J}\mathbf{v} = \\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}}\mathbf{v}

                Since forming sensitivity matrix is very expensive in that this monster is "big" and "dense" matrix!!


        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.muMap * (m)
        dmudm = self.muDeriv
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * \
            self.mesh.aveF2CC.T * sdiag(vol * 1. / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        dCdm_A = Div * (sdiag(Div.T * u) * dMfMuI * dmudm)
        dCdm_RHS1 = Div * (sdiag(self.MfMu0 * B0) * dMfMuI)
        # temp1 = (Dface * (self._Pout.T * self.Bbc_const * self.Bbc))
        # dCdm_RHS2v = (sdiag(vol) * temp1) * \
        #    np.inner(vol, dchidmu * dmudm * v)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmudm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / dCdu.diagonal())
        )
        sol, info = sp.linalg.bicgstab(dCdu, dCdm_v,
                                       tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jvec)")
            # raise Exception ("Iterative solver did not work well")

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu

        dudm = -sol
        dBdmv = (
            sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm * v))
            - sdiag(Div.T * u) * (dMfMuI * (dmudm * v))
            - self.MfMuI * (Div.T * (dudm))
        )

        return mkvc(P * dBdmv)

    @utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Computing Jacobian^T multiplied by vector.

        .. math ::

            (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} = \left[ \mathbf{P}_{deriv}\\frac{\partial \mathbf{\mu} } {\partial \mathbf{m} }
            \left[
            \diag(\M^f_{\mu_{0}^{-1} } \mathbf{B}_0) \dMfMuI  \\
             -  \diag (\Div^T\mathbf{u})\dMfMuI
            \\right ]\\right]^{T}

             -  \left[\mathbf{P}_{deriv}(\MfMui)^{-1}\Div^T\\frac{\delta\mathbf{u}}{\delta \mathbf{m}} \\right]^{T}

        where

        .. math ::

            \mathbf{P}_{derv} = \\frac{\partial \mathbf{P}}{\partial\mathbf{B}}

        .. note ::

            Here we only want to compute

            .. math ::

                \mathbf{J}^{T}\mathbf{v} = (\\frac{\delta \mathbf{P}\mathbf{B}} {\delta \mathbf{m}})^{T} \mathbf{v}

        """
        if u is None:
            u = self.fields(m)

        B, u = u['B'], u['u']
        mu = self.mapping * (m)
        dmudm = self.mapping.deriv(m)
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        Dface = self.mesh.faceDiv
        P = self.survey.projectFieldsDeriv(
            B)                 # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * \
            self.mesh.aveF2CC.T * sdiag(vol * 1. / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)
        s = Div * (self.MfMuI.T * (P.T * v))

        m1 = sp.linalg.interface.aslinearoperator(
            sdiag(1 / (dCdu.T).diagonal())
        )
        sol, info = sp.linalg.bicgstab(dCdu.T, s, tol=1e-6, maxiter=1000, M=m1)

        if info > 0:
            print("Iterative solver did not work well (Jtvec)")
            # raise Exception ("Iterative solver did not work well")

        # dCdm_A = Div * ( sdiag( Div.T * u )* dMfMuI *dmudm  )
        # dCdm_Atsol = ( dMfMuI.T*( sdiag( Div.T * u ) * (Div.T * dmudm)) ) * sol
        dCdm_Atsol = (dmudm.T * dMfMuI.T *
                      (sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( sdiag( self.MfMu0*B0  ) ) * Div.T * dmudm) * sol
        dCdm_RHS1tsol = (
            dmudm.T * dMfMuI.T *
            (sdiag(self.MfMu0 * B0)) * Div.T
        ) * sol

        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        # temp1sol = (Dface.T * (sdiag(vol) * sol))
        # temp2 = self.Bbc_const * (self._Pout.T * self.Bbc).T
        # dCdm_RHS2v  = (sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmudm*v)
        # dCdm_RHS2tsol = (dmudm.T * dchidmu.T * vol) * np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmudm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = sdiag(self.MfMu0 * B0) * (dMfMuI * (dmudm))
        Btemp = sdiag(Div.T * u) * (dMfMuI * (dmudm))
        Jtv = Atemp.T * (P.T * v) - Btemp.T * (P.T * v) - Ctv

        return mkvc(Jtv)


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import (
        Optimization, Regularization,
        Parameters, ObjFunction, Inversion
    )
    prob = MagneticsDiffSecondary(mesh, model)

    miter = kwargs.get('maxIter', 10)

    if prob.ispaired:
        prob.unpair()
    if data.ispaired:
        data.unpair()
    prob.pair(data)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag='D')
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(data, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg


def get_dist_wgt(mesh, receiver_locations, actv, R, R0):
    """
    get_dist_wgt(xn,yn,zn,receiver_locations,R,R0)

    Function creating a distance weighting function required for the magnetic
    inverse problem.

    INPUT
    xn, yn, zn : Node location
    receiver_locations       : Observation locations [obsx, obsy, obsz]
    actv        : Active cell vector [0:air , 1: ground]
    R           : Decay factor (mag=3, grav =2)
    R0          : Small factor added (default=dx/4)

    OUTPUT
    wr       : [nC] Vector of distance weighting

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    # Find non-zero cells
    if actv.dtype == 'bool':
        inds = np.asarray(
            [
                inds for inds, elem in enumerate(actv, 1) if elem
            ],
            dtype=int
        ) - 1
    else:
        inds = actv

    nC = len(inds)

    # Create active cell projector
    P = csr((np.ones(nC), (inds, range(nC))),
                      shape=(mesh.nC, nC))

    # Geometrical constant
    p = 1 / np.sqrt(3)

    # Create cell center location
    Ym, Xm, Zm = np.meshgrid(mesh.vectorCCy, mesh.vectorCCx, mesh.vectorCCz)
    hY, hX, hZ = np.meshgrid(mesh.hy, mesh.hx, mesh.hz)

    # Remove air cells
    Xm = P.T * mkvc(Xm)
    Ym = P.T * mkvc(Ym)
    Zm = P.T * mkvc(Zm)

    hX = P.T * mkvc(hX)
    hY = P.T * mkvc(hY)
    hZ = P.T * mkvc(hZ)

    V = P.T * mkvc(mesh.vol)
    wr = np.zeros(nC)

    ndata = receiver_locations.shape[0]
    count = -1
    print("Begin calculation of distance weighting for R= " + str(R))

    for dd in range(ndata):

        nx1 = (Xm - hX * p - receiver_locations[dd, 0])**2
        nx2 = (Xm + hX * p - receiver_locations[dd, 0])**2

        ny1 = (Ym - hY * p - receiver_locations[dd, 1])**2
        ny2 = (Ym + hY * p - receiver_locations[dd, 1])**2

        nz1 = (Zm - hZ * p - receiver_locations[dd, 2])**2
        nz2 = (Zm + hZ * p - receiver_locations[dd, 2])**2

        R1 = np.sqrt(nx1 + ny1 + nz1)
        R2 = np.sqrt(nx1 + ny1 + nz2)
        R3 = np.sqrt(nx2 + ny1 + nz1)
        R4 = np.sqrt(nx2 + ny1 + nz2)
        R5 = np.sqrt(nx1 + ny2 + nz1)
        R6 = np.sqrt(nx1 + ny2 + nz2)
        R7 = np.sqrt(nx2 + ny2 + nz1)
        R8 = np.sqrt(nx2 + ny2 + nz2)

        temp = (R1 + R0)**-R + (R2 + R0)**-R + (R3 + R0)**-R + \
            (R4 + R0)**-R + (R5 + R0)**-R + (R6 + R0)**-R + \
            (R7 + R0)**-R + (R8 + R0)**-R

        wr = wr + (V * temp / 8.)**2.

        count = progress(dd, count, ndata)

    wr = np.sqrt(wr) / V
    wr = mkvc(wr)
    wr = np.sqrt(wr / (np.max(wr)))

    print("Done 100% ...distance weighting completed!!\n")

    return wr
