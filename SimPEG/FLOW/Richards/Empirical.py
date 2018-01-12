from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
from scipy import constants
from SimPEG import Utils, Props


def _get_projections(u):
    """Get the projections for each domain in the pressure head (u)"""
    nP = len(u)
    bools = u >= 0
    ind_p = np.where(bools)[0]
    ind_n = np.where(~bools)[0]
    P_p = sp.csr_matrix((np.ones(len(ind_p)), (ind_p, ind_p)), shape=(nP, nP))
    P_n = sp.csr_matrix((np.ones(len(ind_n)), (ind_n, ind_n)), shape=(nP, nP))
    return P_p, P_n


def _partition_args(mesh, Hcond, Theta, hcond_args, theta_args, **kwargs):

    hcond_params = {k: kwargs[k] for k in kwargs if k in hcond_args}
    theta_params = {k: kwargs[k] for k in kwargs if k in theta_args}

    other_params = {
        k: kwargs[k] for k in kwargs if k not in hcond_args + theta_args
    }

    if len(other_params) > 0:
        raise Exception('Unknown parameters: {}'.format(other_params))

    hcond = Hcond(mesh, **hcond_params)
    theta = Theta(mesh, **theta_params)

    return hcond, theta


class NonLinearModel(Props.HasModel):
    """A non linear model that has dependence on the fields and a model"""

    counter = None   #: A SimPEG.Utils.Counter object
    mesh = None      #: A SimPEG Mesh

    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super(NonLinearModel, self).__init__(**kwargs)

    @property
    def nP(self):
        """Number of parameters in the model."""
        return self.mesh.nC


class BaseWaterRetention(NonLinearModel):

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        self.validate()

        h = -np.logspace(-2, 3, 1000)
        ax.semilogx(-h, self(h))
        ax.set_title('Water retention curve')
        ax.set_xlabel('Soil water potential, $- \psi$')
        ax.set_ylabel('Water content, $\\theta$')


class BaseHydraulicConductivity(NonLinearModel):

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        self.validate()

        h = -np.logspace(-2, 3, 1000)
        ax.loglog(-h, self(h))
        ax.set_title('Hydraulic conductivity function')
        ax.set_xlabel('Soil water potential, $- \psi$')
        ax.set_ylabel('Hydraulic conductivity, $K$')


class Haverkamp_theta(BaseWaterRetention):

    theta_r, theta_rMap, theta_rDeriv = Props.Invertible(
        "residual water content [L3L-3]",
        default=0.075
    )

    theta_s, theta_sMap, theta_sDeriv = Props.Invertible(
        "saturated water content [L3L-3]",
        default=0.287
    )

    alpha, alphaMap, alphaDeriv = Props.Invertible(
        "",
        default=1.611e+06
    )

    beta, betaMap, betaDeriv = Props.Invertible(
        "",
        default=3.96
    )

    def _get_params(self):
        return self.theta_r, self.theta_s, self.alpha, self.beta

    def __call__(self, u):
        theta_r, theta_s, alpha, beta = self._get_params()

        f = (
            alpha *
            (theta_s - theta_r) /
            (alpha + abs(u)**beta) +
            theta_r
        )

        if np.isscalar(theta_s):
            f[u >= 0] = theta_s
        else:
            f[u >= 0] = theta_s[u >= 0]
        return f

    def derivM(self, u):
        """derivative with respect to m

        .. code::

            import sympy as sy

            alpha, u, beta, theta_r, theta_s = sy.symbols(
                'alpha u beta theta_r theta_s', real=True
            )

            f_n = (
                alpha *
                (theta_s - theta_r) /
                (alpha + abs(u)**beta) +
                theta_r
            )
        """
        return (
            self._derivTheta_r(u) +
            self._derivTheta_s(u) +
            self._derivAlpha(u) +
            self._derivBeta(u)
        )

    def _derivTheta_r(self, u):
        if self.theta_rMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, beta = self._get_params()
        ddm = -alpha/(alpha + abs(u)**beta) + 1
        ddm[u >= 0] = 0
        dT = Utils.sdiag(ddm) * self.theta_rDeriv
        return dT

    def _derivTheta_s(self, u):
        if self.theta_sMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, beta = self._get_params()
        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains
        dT_p = P_p * self.theta_sDeriv
        dT_n = P_n * Utils.sdiag(
            alpha/(alpha + abs(u)**beta)
        ) * self.theta_sDeriv
        return dT_p + dT_n

    def _derivAlpha(self, u):
        if self.alphaMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, beta = self._get_params()
        ddm = -alpha*(-theta_r + theta_s)/(alpha + abs(u)**beta)**2 + (-theta_r + theta_s)/(alpha + abs(u)**beta)
        ddm[u >= 0] = 0
        dA = Utils.sdiag(ddm) * self.alphaDeriv
        return dA

    def _derivBeta(self, u):
        if self.betaMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, beta = self._get_params()
        ddm = -alpha*(-theta_r + theta_s)*np.log(abs(u))*abs(u)**beta/(alpha + abs(u)**beta)**2
        ddm[u >= 0] = 0
        dN = Utils.sdiag(ddm) * self.betaDeriv
        return dN

    def derivU(self, u):
        theta_r, theta_s, alpha, beta = self._get_params()

        g = (
            alpha * (
                (theta_s - theta_r) /
                (alpha + abs(u)**beta)**2
            ) *
            (-beta * abs(u)**(beta-1) * np.sign(u))
        )
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g


class Haverkamp_k(BaseHydraulicConductivity):

    Ks, KsMap, KsDeriv = Props.Invertible(
        "Saturated hydraulic conductivity",
        default=9.44e-03
    )

    A, AMap, ADeriv = Props.Invertible(
        "fitting parameter",
        default=1.175e+06
    )

    gamma, gammaMap, gammaDeriv = Props.Invertible(
        "fitting parameter",
        default=4.74
    )

    def _get_params(self):
        return self.Ks, self.A, self.gamma

    def __call__(self, u):
        Ks, A, gamma = self._get_params()
        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains
        f_p = P_p * np.ones(len(u)) * Ks  # ensures scalar Ks works
        f_n = P_n * Ks * A / (A + abs(u)**gamma)
        return f_p + f_n

    def derivU(self, u):
        Ks, A, gamma = self._get_params()
        g = -(Ks*A*gamma*abs(u)**(gamma-1)*np.sign(u))/((A+abs(u)**gamma)**2)
        g[u >= 0] = 0
        return Utils.sdiag(g)

    def derivM(self, u):
        return self._derivKs(u) + self._derivA(u) + self._derivGamma(u)

    def _derivKs(self, u):
        if self.KsMap is None:
            return Utils.Zero()

        Ks, A, gamma = self._get_params()
        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains

        dKs_dm_p = P_p * self.KsDeriv
        dKs_dm_n = P_n * Utils.sdiag(A/(A + abs(u)**gamma)) * self.KsDeriv
        return dKs_dm_p + dKs_dm_n

    def _derivA(self, u):
        if self.AMap is None:
            return Utils.Zero()
        Ks, A, gamma = self._get_params()
        ddm = Ks / (A + abs(u)**gamma) - Ks*A/(A + abs(u)**gamma)**2
        ddm[u >= 0] = 0
        dA_dm = Utils.sdiag(ddm) * self.ADeriv
        return dA_dm

    def _derivGamma(self, u):
        if self.gammaMap is None:
            return Utils.Zero()
        Ks, A, gamma = self._get_params()
        ddm = -(A*Ks*np.log(abs(u))*abs(u)**gamma)/(A + abs(u)**gamma)**2
        ddm[u >= 0] = 0
        dGamma_dm = Utils.sdiag(ddm) * self.gammaDeriv
        return dGamma_dm


def haverkamp(mesh, **kwargs):
    return _partition_args(
        mesh,
        Haverkamp_k,
        Haverkamp_theta,
        ['Ks', 'A', 'gamma'],
        ['alpha', 'beta', 'theta_r', 'theta_s'],
        **kwargs
    )


class HaverkampParams(object):
    """Holds some default parameterizations for the Haverkamp model."""

    @property
    def celia1990(self):
        """Parameters used in:

            Celia, Michael A., Efthimios T. Bouloutas, and Rebecca L. Zarba.
            "A general mass-conservative numerical solution for the unsaturated
            flow equation." Water Resources Research 26.7 (1990): 1483-1496.
        """
        return {
            'alpha': 1.611e+06,
            'beta': 3.96,
            'theta_r': 0.075,
            'theta_s': 0.287,
            'Ks': 9.44e-03,
            'A': 1.175e+06,
            'gamma': 4.74
        }


class Vangenuchten_theta(BaseWaterRetention):

    theta_r, theta_rMap, theta_rDeriv = Props.Invertible(
        "residual water content [L3L-3]",
        default=0.078
    )

    theta_s, theta_sMap, theta_sDeriv = Props.Invertible(
        "saturated water content [L3L-3]",
        default=0.430
    )

    n, nMap, nDeriv = Props.Invertible(
        "measure of the pore-size distribution, >1",
        default=1.56
    )

    alpha, alphaMap, alphaDeriv = Props.Invertible(
        "related to the inverse of the air entry suction [L-1], >0",
        default=0.036
    )

    def _get_params(self):
        return self.theta_r, self.theta_s, self.alpha, self.n

    def __call__(self, u):
        theta_r, theta_s, alpha, n = self._get_params()
        f = (
            (
                theta_s - theta_r
            ) /
            (
                (1.0 + abs(alpha * u)**n) ** (1.0 - 1.0 / n)
            ) +
            theta_r
        )
        if np.isscalar(theta_s):
            f[u >= 0] = theta_s
        else:
            f[u >= 0] = theta_s[u >= 0]

        return f

    def derivM(self, u):
        """derivative with respect to m

        .. code::

            import sympy as sy

            alpha, u, n, I, Ks, theta_r, theta_s = sy.symbols(
                'alpha u n I Ks theta_r theta_s', real=True
            )

            m = 1.0 - 1.0/n
            theta_e = 1.0 / ((1.0 + sy.functions.Abs(alpha * u) ** n) ** m)

            f_n = (
                (
                    theta_s - theta_r
                ) /
                (
                    (1.0 + abs(alpha * u)**n) ** (1.0 - 1.0 / n)
                ) +
                theta_r
            )
        """
        return (
            self._derivTheta_r(u) +
            self._derivTheta_s(u) +
            self._derivN(u) +
            self._derivAlpha(u)
        )

    def _derivTheta_r(self, u):
        if self.theta_rMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, n = self._get_params()
        ddm = -(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n) + 1
        ddm[u >= 0] = 0
        dT = Utils.sdiag(ddm) * self.theta_rDeriv
        return dT

    def _derivTheta_s(self, u):
        if self.theta_sMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, n = self._get_params()
        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains
        dT_p = P_p * self.theta_sDeriv
        dT_n = P_n * Utils.sdiag(
            (abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n)
        ) * self.theta_sDeriv
        return dT_p + dT_n

    def _derivN(self, u):
        if self.nMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, n = self._get_params()
        ddm = (-theta_r + theta_s)*((-1.0 + 1.0/n)*np.log(abs(alpha*u))*abs(alpha*u)**n/(abs(alpha*u)**n + 1.0) - 1.0*np.log(abs(alpha*u)**n + 1.0)/n**2)*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n)
        ddm[u >= 0] = 0
        dN = Utils.sdiag(ddm) * self.nDeriv
        return dN

    def _derivAlpha(self, u):
        if self.alphaMap is None:
            return Utils.Zero()
        theta_r, theta_s, alpha, n = self._get_params()
        ddm = n*u*(-1.0 + 1.0/n)*(-theta_r + theta_s)*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n)*abs(alpha*u)**n*np.sign(alpha*u)/((abs(alpha*u)**n + 1.0)*abs(alpha*u))
        ddm[u >= 0] = 0
        dA = Utils.sdiag(ddm) * self.alphaDeriv
        return dA

    def derivU(self, u):
        theta_r, theta_s, alpha, n = self._get_params()
        g = -alpha*n*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1./n - 1)*(theta_r - theta_s)*(abs(alpha*u)**n + 1)**(1./n - 2)
        g[u >= 0] = 0
        g = Utils.sdiag(g)
        return g


class Vangenuchten_k(BaseHydraulicConductivity):

    Ks, KsMap, KsDeriv = Props.Invertible(
        "Saturated hydraulic conductivity",
        default=24.96
    )

    I, IMap, IDeriv = Props.Invertible(
        "",
        default=0.5
    )

    n, nMap, nDeriv = Props.Invertible(
        "measure of the pore-size distribution, >1",
        default=1.56
    )

    alpha, alphaMap, alphaDeriv = Props.Invertible(
        "related to the inverse of the air entry suction [L-1], >0",
        default=0.036
    )

    def _get_params(self):
        alpha = self.alpha
        I = self.I
        n = self.n
        Ks = self.Ks
        m = 1.0 - 1.0/n
        return Ks, alpha, I, n, m

    def __call__(self, u):
        Ks, alpha, I, n, m = self._get_params()

        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains
        theta_e = 1.0 / ((1.0 + abs(alpha * u) ** n) ** m)
        f_p = P_p * np.ones(len(u)) * Ks  # ensures scalar Ks works
        f_n = P_n * Ks * theta_e ** I * (
            (1.0 - (1.0 - theta_e ** (1.0 / m)) ** m) ** 2
        )
        return f_p + f_n

    def derivM(self, u):
        """derivative with respect to m

        .. code::

            import sympy as sy

            alpha, u, n, I, Ks, theta_r, theta_s = sy.symbols(
                'alpha u n I Ks theta_r theta_s', real=True
            )

            m = 1.0 - 1.0/n
            theta_e = 1.0 / ((1.0 + sy.functions.Abs(alpha * u) ** n) ** m)

            f_n = Ks * theta_e ** I * (
                (1.0 - (1.0 - theta_e ** (1.0 / m)) ** m) ** 2
            )

            f_n = (
                (
                    theta_s - theta_r
                ) /
                (
                    (1.0 + abs(alpha * u)**n) ** (1.0 - 1.0 / n)
                ) +
                theta_r
            )
        """
        return (
            self._derivKs(u) +
            self._derivI(u) +
            self._derivN(u) +
            self._derivAlpha(u)
        )

    def _derivKs(self, u):
        if self.KsMap is None:
            return Utils.Zero()

        Ks, alpha, I, n, m = self._get_params()
        P_p, P_n = _get_projections(u)  # Compute the positive/negative domains
        theta_e = 1.0 / ((1.0 + abs(alpha * u) ** n) ** m)
        dKs_dm_p = P_p * self.KsDeriv
        dKs_dm_n = P_n * Utils.sdiag(
            theta_e ** I * ((1.0 - (1.0 - theta_e ** (1.0 / m)) ** m) ** 2)
        ) * self.KsDeriv
        return dKs_dm_p + dKs_dm_n

    def _derivAlpha(self, u):
        if self.alphaMap is None:
            return Utils.Zero()
        Ks, alpha, I, n, m = self._get_params()
        ddm = I*u*n*Ks*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*u)**n + 1)**(1.0/n - 2) - (2*u*n*Ks*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))/(((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))
        ddm[u >= 0] = 0
        dA = Utils.sdiag(ddm) * self.alphaDeriv
        return dA

    def _derivN(self, u):
        if self.nMap is None:
            return Utils.Zero()
        Ks, alpha, I, n, m = self._get_params()
        ddm = 1.0*I*Ks*(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**I*((-1.0 + 1.0/n)*np.log(abs(alpha*u))*abs(alpha*u)**n/(abs(alpha*u)**n + 1.0) - 1.0*np.log(abs(alpha*u)**n + 1.0)/n**2)*(-(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0)**(1.0 - 1.0/n) + 1.0)**2*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n)*(abs(alpha*u)**n + 1.0)**(1.0 - 1.0/n) - 2*Ks*(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**I*(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0)**(1.0 - 1.0/n)*(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n))*(1.0 - 1.0/n)*(1.0*((-1.0 + 1.0/n)*np.log(abs(alpha*u))*abs(alpha*u)**n/(abs(alpha*u)**n + 1.0) - 1.0*np.log(abs(alpha*u)**n + 1.0)/n**2)*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n)*(abs(alpha*u)**n + 1.0)**(1.0 - 1.0/n)/(1.0 - 1.0/n) - 1.0*np.log(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))/(n**2*(1.0 - 1.0/n)**2))/(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0) + 1.0*np.log(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0)/n**2)*(-(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0)**(1.0 - 1.0/n) + 1.0)
        ddm[u >= 0] = 0
        dn = Utils.sdiag(ddm) * self.nDeriv
        return dn

    def _derivI(self, u):
        if self.IMap is None:
            return Utils.Zero()
        Ks, alpha, I, n, m = self._get_params()
        ddm = Ks*(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**I*(-(-(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))**(1.0/(1.0 - 1.0/n)) + 1.0)**(1.0 - 1.0/n) + 1.0)**2*np.log(1.0*(abs(alpha*u)**n + 1.0)**(-1.0 + 1.0/n))
        ddm[u >= 0] = 0
        dI = Utils.sdiag(ddm) * self.IDeriv
        return dI

    def derivU(self, u):
        Ks, alpha, I, n, m = self._get_params()
        ddm = I*alpha*n*Ks*abs(alpha*u)**(n - 1.0)*np.sign(alpha*u)*(1.0/n - 1.0)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**(I - 1)*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)**2*(abs(alpha*u)**n + 1)**(1.0/n - 2) - (2*alpha*n*Ks*abs(alpha*u)**(n - 1)*np.sign(alpha*u)*(1.0/n - 1)*((abs(alpha*u)**n + 1)**(1.0/n - 1))**I*((1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1 - 1.0/n) - 1)*(abs(alpha*u)**n + 1)**(1.0/n - 2))/(((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1) + 1)*(1 - 1.0/((abs(alpha*u)**n + 1)**(1.0/n - 1))**(1.0/(1.0/n - 1)))**(1.0/n))
        ddm[u >= 0] = 0
        g = Utils.sdiag(ddm)
        return g


def van_genuchten(mesh, **kwargs):
    return _partition_args(
        mesh,
        Vangenuchten_k,
        Vangenuchten_theta,
        ['alpha', 'n', 'Ks', 'I'],
        ['alpha', 'n', 'theta_r', 'theta_s'],
        **kwargs
    )


class VanGenuchtenParams(object):
    """The RETC code for quantifying the hydraulic functions of unsaturated
    soils, Van Genuchten, M Th, Leij, F J, Yates, S R

    Table 3: Average values for selected soil water retention and hydraulic
    conductivity parameters for 11 major soil textural groups
    according to Rawls et al. [1982]
    """

    @property
    def sand(self):
        return {
            "theta_r": 0.020, "theta_s": 0.417, "alpha": 0.138*100.,
            "n": 1.592, "Ks": 504.0*constants.centi/constants.day
        }

    @property
    def loamy_sand(self):
        return {
            "theta_r": 0.035, "theta_s": 0.401, "alpha": 0.115*100.,
            "n": 1.474, "Ks": 146.6*constants.centi/constants.day
        }

    @property
    def sandy_loam(self):
        return {
            "theta_r": 0.041, "theta_s": 0.412, "alpha": 0.068*100.,
            "n": 1.322, "Ks": 62.16*constants.centi/constants.day
        }

    @property
    def loam(self):
        return {
            "theta_r": 0.027, "theta_s": 0.434, "alpha": 0.090*100.,
            "n": 1.220, "Ks": 16.32*constants.centi/constants.day
        }

    @property
    def silt_loam(self):
        return {
            "theta_r": 0.015, "theta_s": 0.486, "alpha": 0.048*100.,
            "n": 1.211, "Ks": 31.68*constants.centi/constants.day
        }

    @property
    def sandy_clay_loam(self):
        return {
            "theta_r": 0.068, "theta_s": 0.330, "alpha": 0.036*100.,
            "n": 1.250, "Ks": 10.32*constants.centi/constants.day
        }

    @property
    def clay_loam(self):
        return {
            "theta_r": 0.075, "theta_s": 0.390, "alpha": 0.039*100.,
            "n": 1.194, "Ks": 5.52*constants.centi/constants.day
        }

    @property
    def silty_clay_loam(self):
        return {
            "theta_r": 0.040, "theta_s": 0.432, "alpha": 0.031*100.,
            "n": 1.151, "Ks": 3.60*constants.centi/constants.day
        }

    @property
    def sandy_clay(self):
        return {
            "theta_r": 0.109, "theta_s": 0.321, "alpha": 0.034*100.,
            "n": 1.168, "Ks": 2.88*constants.centi/constants.day
        }

    @property
    def silty_clay(self):
        return {
            "theta_r": 0.056, "theta_s": 0.423, "alpha": 0.029*100.,
            "n": 1.127, "Ks": 2.16*constants.centi/constants.day
        }

    @property
    def clay(self):
        return {
            "theta_r": 0.090, "theta_s": 0.385, "alpha": 0.027*100.,
            "n": 1.131, "Ks": 1.44*constants.centi/constants.day
        }

    # From:
    #   INDIRECT METHODS FOR ESTIMATING THE HYDRAULIC
    #   PROPERTIES OF UNSATURATED SOILS
    # @property
    # def siltLoamGE3(self):
    #     """Soil Index: 3310"""
    #     return {"theta_r": 0.139, "theta_s": 0.394, "alpha": 0.00414, "n": 2.15}
    # @property
    # def yoloLightClayK_WC(self):
    #     """Soil Index: None"""
    #     return {"theta_r": 0.205, "theta_s": 0.499, "alpha": 0.02793, "n": 1.71}
    # @property
    # def yoloLightClayK_H(self):
    #     """Soil Index: None"""
    #     return {"theta_r": 0.205, "theta_s": 0.499, "alpha": 0.02793, "n": 1.71}
    # @property
    # def hygieneSandstone(self):
    #     """Soil Index: 4130"""
    #     return {"theta_r": 0.000, "theta_s": 0.256, "alpha": 0.00562, "n": 3.27}
    # @property
    # def lambcrgClay(self):
    #     """Soil Index: 1003"""
    #     return {"theta_r": 0.000, "theta_s": 0.502, "alpha": 0.140, "n": 1.93}
    # @property
    # def beitNetofaClaySoil(self):
    #     """Soil Index: 1006"""
    #     return {"theta_r": 0.000, "theta_s": 0.447, "alpha": 0.00156, "n": 1.17}
    # @property
    # def shiohotSiltyClay(self):
    #     """Soil Index: 1101"""
    #     return {"theta_r": 0.000, "theta_s": 0.456, "alpha": 183, "n":1.17}
    # @property
    # def siltColumbia(self):
    #     """Soil Index: 2001"""
    #     return {"theta_r": 0.146, "theta_s": 0.397,  "alpha": 0.0145, "n": 1.85}
    # @property
    # def siltMontCenis(self):
    #     """Soil Index: 2002"""
    #     return {"theta_r": 0.000, "theta_s": 0.425, "alpha": 0.0103, "n": 1.34}
    # @property
    # def slateDust(self):
    #     """Soil Index: 2004"""
    #     return {"theta_r": 0.000, "theta_s": 0.498, "alpha": 0.00981, "n": 6.75}
    # @property
    # def weldSiltyClayLoam(self):
    #     """Soil Index: 3001"""
    #     return {"theta_r": 0.159, "theta_s": 0.496, "alpha": 0.0136, "n": 5.45}
    # @property
    # def rideauClayLoam_Wetting(self):
    #     """Soil Index: 3101a"""
    #     return {"theta_r": 0.279, "theta_s": 0.419, "alpha": 0.0661, "n": 1.89}
    # @property
    # def rideauClayLoam_Drying(self):
    #     """Soil Index: 3101b"""
    #     return {"theta_r": 0.290, "theta_s": 0.419, "alpha": 0.0177, "n": 3.18}
    # @property
    # def caribouSiltLoam_Drying(self):
    #     """Soil Index: 3301a"""
    #     return {"theta_r": 0.000, "theta_s": 0.451, "alpha": 0.00845, "n": 1.29}
    # @property
    # def caribouSiltLoam_Wetting(self):
    #     """Soil Index: 3301b"""
    #     return {"theta_r": 0.000, "theta_s": 0.450, "alpha": 0.140, "n": 1.09}
    # @property
    # def grenvilleSiltLoam_Wetting(self):
    #     """Soil Index: 3302a"""
    #     return {"theta_r": 0.013, "theta_s": 0523,  "alpha": 0.0630, "n": 1.24}
    # @property
    # def grenvilleSiltLoam_Drying(self):
    #     """Soil Index: 3302c"""
    #     return {"theta_r": 0.000, "theta_s": 0.488, "alpha": 0.0112, "n": 1.23}
    # @property
    # def touchetSiltLoam(self):
    #     """Soil Index: 3304"""
    #     return {"theta_r": 0.183, "theta_s": 0.498, "alpha": 0.0104, "n": 5.78}
    # @property
    # def gilatLoam(self):
    #     """Soil Index: 3402a"""
    #     return {"theta_r": 0.000, "theta_s": 0.454, "alpha": 0.0291, "n": 1.47}
    # @property
    # def pachapaLoam(self):
    #     """Soil Index: 3403"""
    #     return {"theta_r": 0.000, "theta_s": 0.472, "alpha": 0.00829, "n": 1.62}
    # @property
    # def adelantoLoam(self):
    #     """Soil Index: 3404"""
    #     return {"theta_r": 0.000, "theta_s": 0.444, "alpha": 0.00710, "n": 1.26}
    # @property
    # def indioLoam(self):
    #     """Soil Index: 3405a"""
    #     return {"theta_r": 0.000, "theta_s": 0.507, "alpha": 0.00847, "n": 1.60}
    # @property
    # def guclphLoam(self):
    #     """Soil Index: 3407a"""
    #     return {"theta_r": 0.000, "theta_s": 0.563, "alpha": 0.0275, "n": 1.27}
    # @property
    # def guclphLoam(self):
    #     """Soil Index: 3407b"""
    #     return {"theta_r": 0.236, "theta_s": 0.435, "alpha": 0.0271, "n": 262}
    # @property
    # def rubiconSandyLoam(self):
    #     """Soil Index: 3501a"""
    #     return {"theta_r": 0.000, "theta_s": 0.393,  "alpha": 0.00972, "n": 2.18}
    # @property
    # def rubiconSandyLoam(self):
    #     """Soil Index: 350lb"""
    #     return {"theta_r": 0.000, "theta_s": 0.433, "alpha": 0.147, "n": 1.28}
    # @property
    # def pachapaFmeSandyClay(self):
    #     """Soil Index: 3503a"""
    #     return {"theta_r": 0.000, "theta_s": 0.340, "alpha": 0.0194, "n": 1.45}
    # @property
    # def gilatSandyLoam(self):
    #     """Soil Index: 3504"""
    #     return {"theta_r": 0.000, "theta_s": 0.432, "alpha": 0.0103, "n": 1.48}
    # @property
    # def plainfieldSand_210to250(self):
    #     """Soil Index: 4101a"""
    #     return {"theta_r": 0.000, "theta_s": 0.351, "alpha": 0.0236, "n": 12.30}
    # @property
    # def plainfieldSand_210to250(self):
    #     """Soil Index: 4101b"""
    #     return {"theta_r": 0.000, "theta_s": 0.312, "alpha": 0.0387, "n": 4.48}
    # @property
    # def plainfieldSand_177to210(self):
    #     """Soil Index: 4102a"""
    #     return {"theta_r": 0.000, "theta_s": 0.361, "alpha": 0.0207, "n": 10.0}
    # @property
    # def plainfieldSand_177to210(self):
    #     """Soil Index: 4102b"""
    #     return {"theta_r": 0.022, "theta_s": 0.309, "alpha": 0.0328, "n": 6.23}
    # @property
    # def plainfieldSand_149to177(self):
    #     """Soil Index: 4103a"""
    #     return {"theta_r": 0.000, "theta_s": 0.387, "alpha": 0.0173, "n": 7.80}
    # @property
    # def plainfieldSand_149to177(self):
    #     """Soil Index: 4103b"""
    #     return {"theta_r": 0.025, "theta_s": 0.321, "alpha": 0.0272, "n": 6.69}
    # @property
    # def plainfieldSand_l25to149(self):
    #     """Soil Index: 4104a"""
    #     return {"theta_r": 0.000, "theta_s": 03770, "alpha": 0.0145, "n": 10.60}
    # @property
    # def plainfieldSand_125to149(self):
    #     """Soil Index: 4104b"""
    #     return {"theta_r": 0.000, "theta_s": 0.342, "alpha": 0.0230, "n": 5.18}
