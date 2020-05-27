import numpy as np
import properties

from ....utils import mkvc
from ...base import BaseEMSimulation
from ....data import Data
from .... import props

from .survey import Survey

from empymod.transform import dlf

try:
    from empymod.transform import get_spline_values as get_dlf_points
except ImportError:
    from empymod.transform import get_dlf_points
from empymod.utils import check_hankel
from ..utils import static_utils


class Simulation1DLayers(BaseEMSimulation):
    """
    1D DC Simulation
    """

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )

    survey = properties.Instance("a DC survey object", Survey, required=True)

    storeJ = properties.Bool("store the sensitivity", default=False)

    data_type = "volt"
    hankel_pts_per_dec = None  # Default: Standard DLF

    # TODO: using 51 filter coefficient could be overkill, use less if possible
    hankel_filter = "key_51_2012"  # Default: Hankel filter

    _Jmatrix = None
    fix_Jmatrix = False

    def __init__(self, **kwargs):
        BaseEMSimulation.__init__(self, **kwargs)
        try:
            ht, htarg = check_hankel(
                "fht", [self.hankel_filter, self.hankel_pts_per_dec], 1
            )
            self.fhtfilt = htarg[0]  # Store filter
            self.hankel_pts_per_dec = htarg[1]  # Store pts_per_dec
        except ValueError:
            arg = {}
            arg["dlf"] = self.hankel_filter
            if self.hankel_pts_per_dec is not None:
                arg["pts_per_dec"] = self.hankel_pts_per_dec
            ht, htarg = check_hankel("dlf", arg, 1)
            self.fhtfilt = htarg["dlf"]  # Store filter
            self.hankel_pts_per_dec = htarg["pts_per_dec"]  # Store pts_per_dec
        self.hankel_filter = self.fhtfilt.name  # Store name
        self.n_filter = self.fhtfilt.base.size

    def fields(self, m):

        if m is not None:
            self.model = m

        if self.verbose:
            print(">> Compute fields")

        # TODO: this for loop can slow down the speed, cythonize below for loop
        T1 = self.rho[self.n_layer - 1] * np.ones_like(self.lambd)
        for ii in range(self.n_layer - 1, 0, -1):
            rho0 = self.rho[ii - 1]
            t0 = self.thicknesses[ii - 1]
            T0 = (T1 + rho0 * np.tanh(self.lambd * t0)) / (
                1.0 + (T1 * np.tanh(self.lambd * t0) / rho0)
            )
            T1 = T0
        PJ = (T0, None, None)
        try:
            voltage = dlf(
                PJ,
                self.lambd,
                self.offset,
                self.fhtfilt,
                self.hankel_pts_per_dec,
                factAng=None,
                ab=33,
            ).real / (2 * np.pi)
        except TypeError:
            voltage = dlf(
                PJ,
                self.lambd,
                self.offset,
                self.fhtfilt,
                self.hankel_pts_per_dec,
                ang_fact=None,
                ab=33,
            ).real / (2 * np.pi)

        # Assume dipole-dipole
        V = voltage.reshape((self.survey.nD, 4), order="F")
        data = V[:, 0] + V[:, 1] - (V[:, 2] + V[:, 3])

        if self.data_type == "apparent_resistivity":
            data /= self.geometric_factor

        return data

    def dpred(self, m=None, f=None):
        """
        Project fields to receiver locations
        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """

        if self.verbose:
            print("Calculating predicted data")

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        return f

    def getJ(self, m, f=None, factor=1e-2):
        """
            Generate Full sensitivity matrix using central difference
        """
        if self._Jmatrix is not None:
            return self._Jmatrix
        else:
            if self.verbose:
                print("Calculating J and storing")
            self.model = m

            # TODO: this makes code quite slow derive analytic sensitivity
            N = self.survey.nD
            M = self.model.size
            Jmatrix = np.zeros((N, M), dtype=float, order="F")
            for ii in range(M):
                m0 = m.copy()
                dm = m[ii] * factor
                m0[ii] = m[ii] - dm * 0.5
                m1 = m.copy()
                m1[ii] = m[ii] + dm * 0.5
                d0 = self.fields(m0)
                d1 = self.fields(m1)
                Jmatrix[:, ii] = (d1 - d0) / (dm)
            self._Jmatrix = Jmatrix
        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        """
            Compute sensitivity matrix (J) and vector (v) product.
        """

        J = self.getJ(m, f=f)
        Jv = mkvc(np.dot(J, v))

        return mkvc(Jv)

    def Jtvec(self, m, v, f=None):
        """
            Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        """

        J = self.getJ(m, f=f)
        Jtv = mkvc(np.dot(J.T, v))

        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = super(Simulation1DLayers, self).deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete

        if self._Jmatrix is not None:
            toDelete += ["_Jmatrix"]
        return toDelete

    @property
    def electrode_separations(self):
        """
            Electrode separations
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_electrode_separations", None) is None:
            self._electrode_separations = static_utils.electrode_separations(
                self.survey
            )
        return self._electrode_separations

    @property
    def offset(self):
        """
            Offset between a current electrode and a potential electrode
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_offset", None) is None:
            r_AM = self.electrode_separations["AM"]
            r_AN = self.electrode_separations["AN"]
            r_BM = self.electrode_separations["BM"]
            r_BN = self.electrode_separations["BM"]
            self._offset = np.r_[r_AM, r_AN, r_BM, r_BN]
        return self._offset

    @property
    def lambd(self):
        """
            Spatial frequency in Hankel domain
            np.sqrt(kx*2 + ky**2) = lamda
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_lambd", None) is None:
            self._lambd = np.empty(
                [self.offset.size, self.n_filter], order="F", dtype=complex
            )
            self.lambd[:, :], _ = get_dlf_points(
                self.fhtfilt, self.offset, self.hankel_pts_per_dec
            )
        return self._lambd

    # @property
    # def t(self):
    #     """
    #         thickness of the layer
    #     """
    #     # TODO: only works isotropic sigma
    #     if getattr(self, '_t', None) is None:
    #         self._t = self.mesh.hx[:-1]
    #     return self._t

    @property
    def n_layer(self):
        """
            number of layers
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_n_layer", None) is None:
            self._n_layer = self.thicknesses.size + 1
        return self._n_layer

    @property
    def geometric_factor(self):
        """
            number of layers
        """
        # TODO: only works isotropic sigma
        if getattr(self, "_geometric_factor", None) is None:
            r_AM = self.electrode_separations["AM"]
            r_AN = self.electrode_separations["AN"]
            r_BM = self.electrode_separations["BM"]
            r_BN = self.electrode_separations["BM"]
            self._geometric_factor = (1 / r_AM - 1 / r_BM - 1 / r_AN + 1 / r_BN) / (
                2 * np.pi
            )
        return self._geometric_factor
