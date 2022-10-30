import numpy as np

from ....utils import mkvc
from ....simulation import BaseSimulation
from .... import props

from .survey import Survey

from empymod.transform import dlf

try:
    from empymod.transform import get_spline_values as get_dlf_points
except ImportError:
    from empymod.transform import get_dlf_points
from empymod.utils import check_hankel
from ..utils import static_utils
from ....utils import validate_type, validate_string


class Simulation1DLayers(BaseSimulation):
    """
    1D DC Simulation
    """

    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )

    def __init__(
        self,
        survey=None,
        sigma=None,
        sigmaMap=None,
        rho=None,
        rhoMap=None,
        thicknesses=None,
        thicknessesMap=None,
        storeJ=False,
        data_type="volt",
        hankel_pts_per_dec=None,
        hankel_filter="key_51_2012",
        fix_Jmatrix=False,
        **kwargs,
    ):
        super().__init__(survey=survey, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.thicknesses = thicknesses
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap
        self.thicknessesMap = thicknessesMap
        self.storeJ = storeJ
        self.data_type = data_type
        self.fix_Jmatrix = fix_Jmatrix
        try:
            ht, htarg = check_hankel("fht", [hankel_filter, hankel_pts_per_dec], 1)
            self._fhtfilt = htarg[0]  # Store filter
            self._hankel_pts_per_dec = htarg[1]  # Store pts_per_dec
        except ValueError:
            arg = {}
            arg["dlf"] = hankel_filter
            if hankel_pts_per_dec is not None:
                arg["pts_per_dec"] = hankel_pts_per_dec
            ht, htarg = check_hankel("dlf", arg, 1)
            self._fhtfilt = htarg["dlf"]  # Store filter
            self._hankel_pts_per_dec = htarg["pts_per_dec"]  # Store pts_per_dec
        self._hankel_filter = self._fhtfilt.name

    @property
    def survey(self):
        """The DC survey object.

        Returns
        -------
        SimPEG.electromagnetics.static.resistivity.survey.Survey
        """
        if self._survey is None:
            raise AttributeError("Simulation must have a survey.")
        return self._survey

    @survey.setter
    def survey(self, value):
        if value is not None:
            value = validate_type("survey", value, Survey, cast=False)
        self._survey = value

    @property
    def storeJ(self):
        """Whether to store the sensitivity matrix.

        Returns
        -------
        bool
        """
        return self._storeJ

    @storeJ.setter
    def storeJ(self, value):
        self._storeJ = validate_type("storeJ", value, bool)

    @property
    def hankel_filter(self):
        """The hankel filter key.

        Returns
        -------
        str
        """
        return self._hankel_filter

    @property
    def hankel_pts_per_dec(self):
        """Number of hankel transform points per decade.

        Returns
        -------
        int
        """
        return self._hankel_pts_per_dec

    @property
    def fix_Jmatrix(self):
        """Whether to fix the sensitivity matrix between iterations.

        Returns
        -------
        bool
        """
        return self._fix_Jmatrix

    @fix_Jmatrix.setter
    def fix_Jmatrix(self, value):
        self._fix_Jmatrix = validate_type("fix_Jmatrix", value, bool)

    @property
    def data_type(self):
        """The type of data observered by the receivers.

        Returns
        -------
        {"volt", "apparent_resistivity"}
        """
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = validate_string(
            "data_type", value, ["volt", "apparent_resistivity"]
        )

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
            voltage = (
                dlf(
                    PJ,
                    self.lambd,
                    self.offset,
                    self._fhtfilt,
                    self.hankel_pts_per_dec,
                    factAng=None,
                    ab=33,
                ).real
                / (2 * np.pi)
            )
        except TypeError:
            voltage = (
                dlf(
                    PJ,
                    self.lambd,
                    self.offset,
                    self._fhtfilt,
                    self.hankel_pts_per_dec,
                    ang_fact=None,
                    ab=33,
                ).real
                / (2 * np.pi)
            )

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
        if getattr(self, "_Jmatrix", None) is None:
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
        toDelete = super().deleteTheseOnModelUpdate
        if self.fix_Jmatrix:
            return toDelete
        return toDelete + ["_Jmatrix"]

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
                [self.offset.size, self._fhtfilt.base.size], order="F", dtype=complex
            )
            self.lambd[:, :], _ = get_dlf_points(
                self._fhtfilt, self.offset, self.hankel_pts_per_dec
            )
        return self._lambd

    # @property
    # def t(self):
    #     """
    #         thickness of the layer
    #     """
    #     # TODO: only works isotropic sigma
    #     if getattr(self, '_t', None) is None:
    #         self._t = self.mesh.h[0][:-1]
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
