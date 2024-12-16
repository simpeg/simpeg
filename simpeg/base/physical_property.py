import warnings
import numpy as np
from scipy.constants import mu_0
from simpeg import props


class ElectricalConductivity(props.HasModel):
    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    def __init__(self, sigma=None, sigmaMap=None, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap


class MagneticPermeability(props.HasModel):
    mu, muMap, muDeriv = props.Invertible(
        "Magnetic Permeability (H/m)",
    )
    mui, muiMap, muiDeriv = props.Invertible("Inverse Magnetic Permeability (m/H)")
    props.Reciprocal(mu, mui)

    def __init__(self, mu=mu_0, muMap=None, mui=None, muiMap=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.mui = mui
        self.muMap = muMap
        self.muiMap = muiMap


class DielectricPermittivity(props.HasModel):
    permittivity = props.PhysicalProperty(
        "Dielectric permittivity (F/m)", optional=True
    )

    def __init__(self, permittivity=None, **kwargs):

        if permittivity is not None:
            warnings.warn(
                "Simulations using permittivity have not yet been thoroughly tested and derivatives are not implemented. Contributions welcome!",
                UserWarning,
                stacklevel=3,
            )
        self.permittivity = permittivity
        super().__init__(**kwargs)


class MassDensity(props.HasModel):

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)")

    def __init__(self, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.rhoMap = rhoMap


class MagneticSusceptibility(props.HasModel):

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(self, chi=None, chiMap=None, **kwargs):
        super().__init__(**kwargs)
        self.chi = chi
        self.chiMap = chiMap


class LayerThickness(props.HasModel):
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)"
    )

    def __init__(self, thicknesses=None, thicknessesMap=None, **kwargs):
        super().__init__(**kwargs)
        if thicknesses is None:
            thicknesses = []
        self.thicknesses = thicknesses
        self.thicknessesMap = thicknessesMap


class ElectricalChargeability(props.HasModel):
    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    def __init__(self, eta=None, etaMap=None, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.etaMap = etaMap


class AcousticVelocity(props.HasModel):
    slowness, slownessMap, slownessDeriv = props.Invertible("Slowness model (s/m)")
    velocity, velocityMap, velocityDeriv = props.Invertible("Velocity model (m/s)")
    props.Reciprocal(slowness, velocity)

    def __init__(
        self, slowness=None, slownessMap=None, velocity=None, velocityMap=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.slowness = slowness
        self.slownessMap = slownessMap
        self.velocity = velocity
        self.velocityMap = velocityMap


class HydraulicConductivity(props.HasModel):
    Ks, KsMap, KsDeriv = props.Invertible("Saturated hydraulic conductivity")

    def __init__(self, Ks=None, KsMap=None, **kwargs):
        self.Ks = Ks
        self.KsMap = KsMap
        super().__init__(**kwargs)

    # Ideally this becomes an abstract method once the PropertyMeta inherits from
    # ABCMeta
    def __call__(self, value):
        raise NotImplementedError("__call__ has not been implemented.")

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        h = -np.logspace(-2, 3, 1000)
        ax.loglog(-h, self(h))
        ax.set_title("Hydraulic conductivity function")
        ax.set_xlabel(r"Soil water potential, $-\psi$")
        ax.set_ylabel("Hydraulic conductivity, $K$")


class WaterRetention(props.HasModel):

    theta_r, theta_rMap, theta_rDeriv = props.Invertible(
        "residual water content [L3L-3]"
    )

    theta_s, theta_sMap, theta_sDeriv = props.Invertible(
        "saturated water content [L3L-3]"
    )

    def __init__(
        self,
        theta_r=None,
        theta_rMap=None,
        theta_s=None,
        theta_sMap=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta_r = theta_r
        self.theta_rMap = theta_rMap
        self.theta_s = theta_s
        self.theta_sMap = theta_sMap

    # Ideally this becomes an abstract method once the PropertyMeta inherits from
    # ABCMeta
    def __call__(self, value):
        raise NotImplementedError("__call__ has not been implemented.")

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)

        h = -np.logspace(-2, 3, 1000)
        ax.semilogx(-h, self(h))
        ax.set_title("Water retention curve")
        ax.set_xlabel(r"Soil water potential, $-\psi$")
        ax.set_ylabel("Water content, $\\theta$")
