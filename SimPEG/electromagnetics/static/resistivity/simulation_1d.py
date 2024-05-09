import numpy as np

from ....simulation import BaseSimulation
from .... import props

from .survey import Survey

from empymod.transform import get_dlf_points
from empymod import filters
from ....utils import validate_type, validate_string
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline


HANKEL_FILTERS = [
    "kong_61_2007",
    "kong_241_2007",
    "key_101_2009",
    "key_201_2009",
    "key_401_2009",
    "anderson_801_1982",
    "key_51_2012",
    "key_101_2012",
    "key_201_2012",
    "wer_201_2018",
]


def _phi_tilde(rho, thicknesses, lambdas):
    """Calculate potential in the hankel domain.

    Parameters
    ----------
    rho : (n_layer) np.ndarray
        Resistivity array (ohm m)
    thicknesses : (n_layer - 1) np.ndarray
        Array of layer thicknesses defined from the top down (m).
    lambdas : (n_lambda) np.ndarray
        Wavenumbers.

    Returns
    -------
    (n_lambda) np.ndarray
        Potential in wavenumber domain at sampled locations
    """
    n_layer = len(rho)
    tanh = np.tanh(lambdas[None, :] * thicknesses[:, None])
    t = rho[-1] * np.ones_like(lambdas)
    for i in range(n_layer - 2, -1, -1):
        t = (t + rho[i] * tanh[i]) / (1.0 + t * tanh[i] / rho[i])
    return t


def _dphi_tilde(rho, thicknesses, lambdas):
    """Calculate derivative of potential in the hankel domain.

    Parameters
    ----------
    rho : (n_layer) np.ndarray
        Resistivity array (ohm m)
    thicknesses : (n_layer - 1) np.ndarray
        Array of layer thicknesses defined from the top down (m).
    lambdas : (n_lambda) np.ndarray
        Wavenumbers.

    Returns
    -------
    J_rho : (n_lambda, n_layer) np.ndarray
        Jacobian matrix of first derivatives of lambda w.r.t. resistivity.
    J_h : (n_lambda, n_layer-1) np.ndarray
        Jacobian matrix of first derivatives of lambda w.r.t. thicknesses.
    """
    n_layer = len(rho)
    ts = np.empty((n_layer, len(lambdas)))
    tanh = np.tanh(lambdas[None, :] * thicknesses[:, None])
    tops = np.empty((n_layer, len(lambdas)))
    bots = np.empty((n_layer, len(lambdas)))
    ts[-1] = rho[-1]
    for i in range(n_layer - 2, -1, -1):
        ts[i] = ts[i + 1]
        tops[i] = ts[i + 1] + rho[i] * tanh[i]
        bots[i] = 1 + ts[i + 1] * tanh[i] / rho[i]
        ts[i] = tops[i] / bots[i]
    # return ts[0]
    # ts0 = 1.0
    g_ti = np.ones(len(lambdas))
    J_rho = np.empty((n_layer, len(lambdas)))
    J_h = np.empty((n_layer - 1, len(lambdas)))
    for i in range(n_layer - 1):
        # ts[i] = tops[i] / bots[i]
        g_tops = g_ti / bots[i]
        g_bots = -ts[i] / bots[i] * g_ti
        # bots[i] = 1 + ts[i+1] * tanh[i] / rho0
        g_tip1 = tanh[i] / rho[i] * g_bots
        g_tanh = ts[i + 1] / rho[i] * g_bots
        g_rho0 = -ts[i + 1] * tanh[i] / (rho[i] ** 2) * g_bots
        # tops[i] = ts[i+1] + rho0 * tanh[i]
        g_tip1 += g_tops
        g_tanh += rho[i] * g_tops
        g_rho0 += tanh[i] * g_tops
        # tanh = tanh(thick * lambd)
        g_thick = (1 - tanh[i] ** 2) * lambdas * g_tanh

        J_rho[i] = g_rho0
        J_h[i] = g_thick

        g_ti = g_tip1
    J_rho[-1] = g_ti
    return J_rho.T, J_h.T


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
        hankel_filter="key_201_2012",
        fix_Jmatrix=False,
        **kwargs,
    ):
        if kwargs.pop("storeJ", None) is not None:
            raise TypeError(
                "storeJ is no longer settable by the user for this simulation."
            )
        if kwargs.pop("hankel_pts_per_dec", None) is not None:
            raise TypeError(
                "hankel_pts_per_dec is no longer settable by the user for this simulation."
            )
        if kwargs.pop("data_type", None) is not None:
            raise TypeError(
                "data_type can no longer be set on the simulation, it must be set on each"
                "receiver."
            )
        super().__init__(survey=survey, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.thicknesses = thicknesses
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap
        self.thicknessesMap = thicknessesMap
        self.fix_Jmatrix = fix_Jmatrix
        self.hankel_filter = hankel_filter  # Store filter
        self._coefficients_set = False
        self._storeJ = True

    @property
    def survey(self):
        """The DC survey object.

        Returns
        -------
        simpeg.electromagnetics.static.resistivity.survey.Survey
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

    @property
    def hankel_filter(self):
        """The hankel filter key.

        Returns
        -------
        str
        """
        return self._hankel_filter

    @hankel_filter.setter
    def hankel_filter(self, value):
        self._hankel_filter = validate_string(
            "hankel_filter",
            value,
            HANKEL_FILTERS,
        )
        self._fhtfilt = getattr(filters, self._hankel_filter)()

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

    def _compute_hankel_coefficients(self):
        if self._coefficients_set:
            return
        survey = self.survey

        r_min = np.infty
        r_max = -np.infty

        for src in survey.source_list:
            src_loc = src.location
            for rx in src.receiver_list:
                rx_loc = rx.locations
                if not isinstance(rx_loc, list):
                    # is a pole receiver
                    rx_loc = [rx_loc]
                for loc in rx_loc:
                    off = np.linalg.norm(src_loc[:, None, :] - loc[None, :, :], axis=-1)
                    r_min = min(off.min(), r_min)
                    r_max = max(off.max(), r_max)
        self.survey.set_geometric_factor()

        lambdas, r_spline_points = get_dlf_points(
            self._fhtfilt, np.r_[r_min, r_max], -1
        )
        lambdas = lambdas.reshape(-1)
        n_lambda = len(lambdas)
        n_r = len(r_spline_points)

        n_base = len(self._fhtfilt.base)
        A_dht = np.zeros((n_r, n_lambda))
        for i in range(n_r):
            A_dht[i, i : i + n_base] = self._fhtfilt.j0
        A_dht = A_dht[::-1]  # shuffle these back

        # A_dht goes from wavenumber to space at r_spline_points
        # Then need to spline it from r_spline to all offsets
        # Calculate the interpolating spline basis functions for each spline point
        splines = []
        for i in range(n_r):
            e = np.zeros(n_r)
            e[i] = 1.0
            sp = iuSpline(np.log(r_spline_points[::-1]), e, k=5)
            splines.append(sp)
        # As will go from wavenumber to space domain
        As = []
        for src in survey.source_list:
            src_loc = src.location
            for rx in src.receiver_list:
                rx_loc = rx.locations
                A = np.zeros((rx.nD, n_r))
                for current, tx_elec_loc in zip(src.current, src_loc):
                    if not isinstance(rx_loc, list):
                        # is a pole receiver
                        m_off = np.linalg.norm(tx_elec_loc - rx_loc, axis=-1)
                        n_off = None
                    else:
                        m_off = np.linalg.norm(tx_elec_loc - rx_loc[0], axis=-1)
                        n_off = np.linalg.norm(tx_elec_loc - rx_loc[1], axis=-1)
                    # This receiver has a bunch of data...
                    # this A is the linear operation going from the splined offsets to the data offset
                    for i in range(n_r):
                        A[:, i] += current * splines[i](np.log(m_off)) / m_off
                        if n_off is not None:
                            A[:, i] -= current * splines[i](np.log(n_off)) / n_off
                if rx.data_type == "apparent_resistivity":
                    A /= rx.geometric_factor[src]
                As.append(A @ A_dht / (2 * np.pi))
        self._coefficients_set = True
        self._As = np.vstack(As)
        self._lambdas = lambdas

    def fields(self, m):
        self.model = m
        self._compute_hankel_coefficients()
        return _phi_tilde(self.rho, self.thicknesses, self._lambdas)

    def dpred(self, m=None, f=None):
        """
        Project fields to receiver locations
        :param Fields u: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        if self.verbose:
            print("Calculating predicted data")

        self._compute_hankel_coefficients()
        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)
        return self._As @ f

    def getJ(self, m, f=None, factor=1e-2):
        """
        Generate Full sensitivity matrix using central difference
        """
        self.model = m
        if getattr(self, "_Jmatrix", None) is None:
            self._compute_hankel_coefficients()
            if self.verbose:
                print("Calculating J and storing")

            J_rho, J_h = _dphi_tilde(self.rho, self.thicknesses, self._lambdas)

            Jmatrix = 0
            if self.rhoMap is not None:
                Jmatrix += (self._As @ J_rho) @ self.rhoDeriv
            if self.thicknessesMap is not None:
                Jmatrix += (self._As @ J_h) @ self.thicknessesDeriv
            self._Jmatrix = Jmatrix
        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        """
        Compute sensitivity matrix (J) and vector (v) product.
        """
        return self.getJ(m, f=f) @ v

    def Jtvec(self, m, v, f=None):
        """
        Compute adjoint sensitivity matrix (J^T) and vector (v) product.
        """
        return self.getJ(m, f=f).T @ v

    @property
    def deleteTheseOnModelUpdate(self):
        to_delete = super().deleteTheseOnModelUpdate
        if not self.fix_Jmatrix:
            to_delete = to_delete + ["_Jmatrix"]
        return to_delete
