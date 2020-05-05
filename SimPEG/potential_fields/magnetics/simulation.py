from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from ...utils.code_utils import deprecate_class

from SimPEG import utils
from ...simulation import BaseSimulation
from ..base import BasePFSimulation
from .survey import MagneticSurvey
from .analytics import CongruousMagBC

from SimPEG import Solver
from SimPEG import props
import properties
from SimPEG.utils import mkvc, mat_utils, sdiag, setKwargs


class Simulation3DIntegral(BasePFSimulation):
    """
    magnetic simulation in integral form.

    """

    chi, chiMap, chiDeriv = props.Invertible(
        "Magnetic Susceptibility (SI)",
        default=1.
    )

    modelType = properties.StringChoice(
        "Type of magnetization model",
        choices=['susceptibility', 'vector'],
        default='susceptibility'
    )

    is_amplitude_data = properties.Boolean(
        "Whether the supplied data is amplitude data",
        default=False
    )


    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)
        self._G = None
        self._M = None
        self._gtg_diagonal = None
        self.modelMap = self.chiMap
        setKwargs(self, **kwargs)

    @property
    def M(self):
        """
        M: ndarray
            Magnetization matrix
        """
        if getattr(self, "_M", None) is None:

            if self.modelType == 'vector':
                self._M = sp.identity(self.nC) * self.survey.source_field.parameters[0]

            else:
                mag = mat_utils.dip_azimuth2cartesian(
                    np.ones(self.nC) * self.survey.source_field.parameters[1],
                    np.ones(self.nC) * self.survey.source_field.parameters[2]
                )

                self._M = sp.vstack(
                    (
                        sdiag(mag[:, 0] * self.survey.source_field.parameters[0]),
                        sdiag(mag[:, 1] * self.survey.source_field.parameters[0]),
                        sdiag(mag[:, 2] * self.survey.source_field.parameters[0])
                    )
                )

        return self._M

    @M.setter
    def M(self, M):
        """
        Create magnetization matrix from unit vector orientation
        :parameter
        M: array (3*nC,) or (nC, 3)
        """
        if self.modelType == 'vector':
            self._M = sdiag(mkvc(M)*self.survey.source_field.parameters[0])
        else:
            M = M.reshape((-1, 3))
            self._M = sp.vstack(
                (
                    sdiag(M[:, 0] * self.survey.source_field.parameters[0]),
                    sdiag(M[:, 1] * self.survey.source_field.parameters[0]),
                    sdiag(M[:, 2] * self.survey.source_field.parameters[0])
                )
            )

    def fields(self, model):

        model = self.chiMap * model

        if self.store_sensitivities == 'forward_only':
            self.model = model
            fields = mkvc(self.linear_operator())
        else:
            fields = np.asarray(self.G@model.astype(np.float32))

        if self.is_amplitude_data:
            fields = self.compute_amplitude(fields)

        return fields

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
    def tmi_projection(self):

        if getattr(self, '_tmi_projection', None) is None:

            # Convert from north to cartesian
            self._tmi_projection = mat_utils.dip_azimuth2cartesian(
                self.survey.source_field.parameters[1],
                self.survey.source_field.parameters[2]
            )

        return self._tmi_projection

    def getJtJdiag(self, m, W=None):
        """
            Return the diagonal of JtJ
        """
        self.model = m

        if W is None:
            W = np.ones(self.nD)
        else:
            W = W.diagonal()**2
        if getattr(self, "_gtg_diagonal", None) is None:
            diag = np.zeros(self.G.shape[1])
            if not self.is_amplitude_data:
                for i in range(len(W)):
                    diag += W[i]*(self.G[i]*self.G[i])
            else:
                fieldDeriv = self.fieldDeriv
                Gx = self.G[::3]
                Gy = self.G[1::3]
                Gz = self.G[2::3]
                for i in range(len(W)):
                    row = (fieldDeriv[0, i]*Gx[i] +
                           fieldDeriv[1, i]*Gy[i] +
                           fieldDeriv[2, i]*Gz[i])
                    diag += W[i]*(row*row)
            self._gtg_diagonal = diag
        else:
            diag = self._gtg_diagonal
        return mkvc((sdiag(np.sqrt(diag))@self.chiDeriv).power(2).sum(axis=0))

    def Jvec(self, m, v, f=None):
        if self.chi is None:
            self.model = np.zeros(self.G.shape[1])
        dmu_dm_v = self.chiDeriv@v

        Jvec = self.G@dmu_dm_v.astype(np.float32)

        if self.is_amplitude_data:
            Jvec = Jvec.reshape((-1, 3)).T
            fieldDeriv_Jvec = self.fieldDeriv * Jvec
            return fieldDeriv_Jvec[0] + fieldDeriv_Jvec[1] + fieldDeriv_Jvec[2]
        else:
            return Jvec

    def Jtvec(self, m, v, f=None):
        if self.chi is None:
            self.model = np.zeros(self.G.shape[1])

        if self.is_amplitude_data:
            v = (self.fieldDeriv * v).T.reshape(-1)
        Jtvec = self.G.T@v.astype(np.float32)
        return np.asarray(self.chiDeriv.T@Jtvec)

    @property
    def fieldDeriv(self):

        if self.chi is None:
            self.model = np.zeros(self.G.shape[1])

        if getattr(self, '_fieldDeriv', None) is None:
            fields = np.asarray(self.G.dot((self.chiMap@self.chi).astype(np.float32)))
            b_xyz = self.normalized_fields(fields)

            self._fieldDeriv = b_xyz

        return self._fieldDeriv

    @classmethod
    def normalized_fields(cls, fields):
        """
            Return the normalized B fields
        """

        # Get field amplitude
        amp = cls.compute_amplitude(fields.astype(np.float64))

        return fields.reshape((3, -1), order='F')/amp[None, :]

    @classmethod
    def compute_amplitude(cls, b_xyz):
        """
            Compute amplitude of the magnetic field
        """

        amplitude = np.linalg.norm(b_xyz.reshape((3, -1), order='F'), axis=0)

        return amplitude

    def evaluate_integral(self, receiver_location, components):
        """
            Load in the active nodes of a tensor mesh and computes the magnetic
            forward relation between a cuboid and a given observation
            location outside the Earth [obsx, obsy, obsz]

            INPUT:
            receiver_location:  [obsx, obsy, obsz] nC x 3 Array

            components: list[str]
                List of magnetic components chosen from:
                'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz'

            OUTPUT:
            Tx = [Txx Txy Txz]
            Ty = [Tyx Tyy Tyz]
            Tz = [Tzx Tzy Tzz]
        """
        # TODO: This should probably be converted to C
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
            rows["byz"][0, nC:2*nC] = (
                    2 * (
                        (((dy2 * arg5) / (r1 * (arg2**2) + (dy2**2) * r1 + eps))) -
                        (((dy2 * arg10) / (r2 * (arg7**2) + (dy2**2) * r2 + eps))) +
                        (((dy2 * arg15) / (r3 * (arg12**2) + (dy2**2) * r3 + eps))) -
                        (((dy2 * arg20) / (r4 * (arg17**2) + (dy2**2) * r4 + eps))) +
                        (((dy1 * arg25) / (r5 * (arg22**2) + (dy1**2) * r5 + eps))) -
                        (((dy1 * arg30) / (r6 * (arg27**2) + (dy1**2) * r6 + eps))) +
                        (((dy1 * arg35) / (r7 * (arg32**2) + (dy1**2) * r7 + eps))) -
                        (((dy1 * arg40) / (r8 * (arg37**2) + (dy1**2) * r8 + eps)))
                )
            )
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
            rows["bx"] /= (-4 * np.pi)

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

            rows["tmi"] = np.dot(self.tmi_projection, np.r_[rows["bx"], rows["by"], rows["bz"]])

        return np.vstack([rows[component] for component in components])

    @property
    def deleteTheseOnModelUpdate(self):
        deletes = super().deleteTheseOnModelUpdate
        if self.is_amplitude_data:
            deletes += ['_gtg_diagonal']
        return deletes


class Simulation3DDifferential(BaseSimulation):
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

    survey = properties.Instance(
            "a survey object", MagneticSurvey, required=True
    )

    def __init__(self, mesh, **kwargs):
        super().__init__(mesh, **kwargs)

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
        b0 = self.survey.source_field.b0
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

        mu = self.muMap * m
        chi = mu / mu_0 - 1

        # Temporary fix
        Bbc, Bbc_const = CongruousMagBC(self.mesh, self.survey.source_field.b0, chi)
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
        Ainv = self.solver(A, **self.solver_opts)
        u = Ainv * rhs
        B0 = self.getB0()
        B = self.MfMuI * self.MfMu0 * B0 - B0 - self.MfMuI * self._Div.T * u
        Ainv.clean()

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
        dmu_dm = self.muDeriv
        # dchidmu = sdiag(1 / mu_0 * np.ones(self.mesh.nC))

        vol = self.mesh.vol
        Div = self._Div
        P = self.survey.projectFieldsDeriv(B)  # Projection matrix
        B0 = self.getB0()

        MfMuIvec = 1 / self.MfMui.diagonal()
        dMfMuI = sdiag(MfMuIvec**2) * \
            self.mesh.aveF2CC.T * sdiag(vol * 1. / mu**2)

        # A = self._Div*self.MfMuI*self._Div.T
        # RHS = Div*MfMuI*MfMu0*B0 - Div*B0 + Mc*Dface*Pout.T*Bbc
        # C(m,u) = A*m-rhs
        # dudm = -(dCdu)^(-1)dCdm

        dCdu = self.getA(m)  # = A
        dCdm_A = Div * (sdiag(Div.T * u) * dMfMuI * dmu_dm)
        dCdm_RHS1 = Div * (sdiag(self.MfMu0 * B0) * dMfMuI)
        # temp1 = (Dface * (self._Pout.T * self.Bbc_const * self.Bbc))
        # dCdm_RHS2v = (sdiag(vol) * temp1) * \
        #    np.inner(vol, dchidmu * dmu_dm * v)

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v
        dCdm_RHSv = dCdm_RHS1 * (dmu_dm * v)
        dCdm_v = dCdm_A * v - dCdm_RHSv

        Ainv = self.solver(dCdu, **self.solver_opts)
        sol = Ainv * dCdm_v

        dudm = -sol
        dBdmv = (
            sdiag(self.MfMu0 * B0) * (dMfMuI * (dmu_dm * v))
            - sdiag(Div.T * u) * (dMfMuI * (dmu_dm * v))
            - self.MfMuI * (Div.T * (dudm))
        )

        Ainv.clean()

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
        dmu_dm = self.mapping.deriv(m)
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

        Ainv = self.solver(dCdu.T, **self.solver_opts)
        sol = Ainv * s

        Ainv.clean()

        # dCdm_A = Div * ( sdiag( Div.T * u )* dMfMuI *dmu_dm  )
        # dCdm_Atsol = ( dMfMuI.T*( sdiag( Div.T * u ) * (Div.T * dmu_dm)) ) * sol
        dCdm_Atsol = (dmu_dm.T * dMfMuI.T *
                      (sdiag(Div.T * u) * Div.T)) * sol

        # dCdm_RHS1 = Div * (sdiag( self.MfMu0*B0  ) * dMfMuI)
        # dCdm_RHS1tsol = (dMfMuI.T*( sdiag( self.MfMu0*B0  ) ) * Div.T * dmu_dm) * sol
        dCdm_RHS1tsol = (
            dmu_dm.T * dMfMuI.T *
            (sdiag(self.MfMu0 * B0)) * Div.T
        ) * sol

        # temp1 = (Dface*(self._Pout.T*self.Bbc_const*self.Bbc))
        # temp1sol = (Dface.T * (sdiag(vol) * sol))
        # temp2 = self.Bbc_const * (self._Pout.T * self.Bbc).T
        # dCdm_RHS2v  = (sdiag(vol)*temp1)*np.inner(vol, dchidmu*dmu_dm*v)
        # dCdm_RHS2tsol = (dmu_dm.T * dchidmu.T * vol) * np.inner(temp2, temp1sol)

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v

        # temporary fix
        # dCdm_RHStsol = dCdm_RHS1tsol - dCdm_RHS2tsol
        dCdm_RHStsol = dCdm_RHS1tsol

        # dCdm_RHSv =  dCdm_RHS1*(dmu_dm*v) +  dCdm_RHS2v
        # dCdm_v = dCdm_A*v - dCdm_RHSv

        Ctv = dCdm_Atsol - dCdm_RHStsol

        # B = self.MfMuI*self.MfMu0*B0-B0-self.MfMuI*self._Div.T*u
        # dBdm = d\mudm*dBd\mu
        # dPBdm^T*v = Atemp^T*P^T*v - Btemp^T*P^T*v - Ctv

        Atemp = sdiag(self.MfMu0 * B0) * (dMfMuI * (dmu_dm))
        Btemp = sdiag(Div.T * u) * (dMfMuI * (dmu_dm))
        Jtv = Atemp.T * (P.T * v) - Btemp.T * (P.T * v) - Ctv

        return mkvc(Jtv)

    @property
    def Qfx(self):
        if getattr(self, '_Qfx', None) is None:
            self._Qfx = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, 'Fx'
            )
        return self._Qfx

    @property
    def Qfy(self):
        if getattr(self, '_Qfy', None) is None:
            self._Qfy = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, 'Fy'
            )
        return self._Qfy

    @property
    def Qfz(self):
        if getattr(self, '_Qfz', None) is None:
            self._Qfz = self.mesh.getInterpolationMat(
                self.survey.receiver_locations, 'Fz'
            )
        return self._Qfz

    def projectFields(self, u):
        """
            This function projects the fields onto the data space.
            Especially, here for we use total magnetic intensity (TMI) data,
            which is common in practice.
            First we project our B on to data location

            .. math::

                \mathbf{B}_{rec} = \mathbf{P} \mathbf{B}

            then we take the dot product between B and b_0

            .. math ::

                \\text{TMI} = \\vec{B}_s \cdot \hat{B}_0

        """
        # TODO: There can be some different tyes of data like |B| or B
        components = self.survey.components

        fields = {}
        if 'bx' in components or 'tmi' in components:
            fields['bx'] = self.Qfx * u['B']
        if 'by' in components or 'tmi' in components:
            fields['by'] = self.Qfy * u['B']
        if 'bz' in components or 'tmi' in components:
            fields['bz'] = self.Qfz * u['B']

        if 'tmi' in components:
            bx = fields['bx']
            by = fields['by']
            bz = fields['bz']
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0]**2 + B0[1]**2 + B0[2]**2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields['tmi'] = bx * box + by * boy + bz * boz

        return np.concatenate([fields[comp] for comp in components])

    @utils.count
    def projectFieldsDeriv(self, B):
        """
            This function projects the fields onto the data space.

            .. math::

                \\frac{\partial d_\\text{pred}}{\partial \mathbf{B}} = \mathbf{P}

            Especially, this function is for TMI data type
        """

        components = self.survey.components

        fields = {}
        if 'bx' in components or 'tmi' in components:
            fields['bx'] = self.Qfx
        if 'by' in components or 'tmi' in components:
            fields['by'] = self.Qfy
        if 'bz' in components or 'tmi' in components:
            fields['bz'] = self.Qfz

        if 'tmi' in components:
            bx = fields['bx']
            by = fields['by']
            bz = fields['bz']
            # Generate unit vector
            B0 = self.survey.source_field.b0
            Bot = np.sqrt(B0[0]**2 + B0[1]**2 + B0[2]**2)
            box = B0[0] / Bot
            boy = B0[1] / Bot
            boz = B0[2] / Bot
            fields['tmi'] = bx * box + by * boy + bz * boz

        return sp.vstack([fields[comp] for comp in components])

    def projectFieldsAsVector(self, B):

        bfx = self.Qfx * B
        bfy = self.Qfy * B
        bfz = self.Qfz * B

        return np.r_[bfx, bfy, bfz]


def MagneticsDiffSecondaryInv(mesh, model, data, **kwargs):
    """
        Inversion module for MagneticsDiffSecondary

    """
    from SimPEG import (
        Optimization, Regularization,
        Parameters, ObjFunction, Inversion
    )
    prob = Simulation3DDifferential(
           mesh,
           survey=data,
           mu=model)

    miter = kwargs.get('maxIter', 10)

    # Create an optimization program
    opt = Optimization.InexactGaussNewton(maxIter=miter)
    opt.bfgsH0 = Solver(sp.identity(model.nP), flag='D')
    # Create a regularization program
    reg = Regularization.Tikhonov(model)
    # Create an objective function
    beta = Parameters.BetaSchedule(beta0=1e0)
    obj = ObjFunction.BaseObjFunction(prob, reg, beta=beta)
    # Create an inversion object
    inv = Inversion.BaseInversion(obj, opt)

    return inv, reg


############
# Deprecated
############

@deprecate_class(removal_version='0.15.0')
class MagneticIntegral(Simulation3DIntegral):
    pass


@deprecate_class(removal_version='0.15.0')
class Problem3D_Diff(Simulation3DDifferential):
    pass
