import warnings
import numpy as np
from scipy.constants import mu_0
from simpeg import props


@props._add_deprecated_physical_property_functions("sigma")
@props._add_deprecated_physical_property_functions("rho")
class ElectricalConductivity(props.HasModel):
    sigma = props.PhysicalProperty("Electrical conductivity (S/m)")
    rho = props.PhysicalProperty("Electrical resistivity (Ohm m)")
    sigma.set_reciprocal(rho)

    def __init__(self, sigma=None, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho


@props._add_deprecated_physical_property_functions("mu")
@props._add_deprecated_physical_property_functions("mui")
class MagneticPermeability(props.HasModel):
    mu = props.PhysicalProperty("Magnetic Permeability (H/m)")
    mui = props.PhysicalProperty("Inverse Magnetic Permeability (m/H)")
    mu.set_reciprocal(mui)

    def __init__(self, mu=mu_0, mui=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.mui = mui


class DielectricPermittivity(props.HasModel):
    permittivity = props.PhysicalProperty(
        "Dielectric permittivity (F/m)", default=None, invertible=False
    )

    def __init__(self, permittivity=None, **kwargs):

        if permittivity is not None:
            warnings.warn(
                "Simulations using permittivity have not yet been thoroughly tested and derivatives are not implemented. Contributions welcome!",
                stacklevel=3,
            )
        self.permittivity = permittivity
        super().__init__(**kwargs)


@props._add_deprecated_physical_property_functions("rho")
class MassDensity(props.HasModel):

    rho = props.PhysicalProperty("Specific density (g/cc)")

    def __init__(self, rho=None, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho


@props._add_deprecated_physical_property_functions("chi")
class MagneticSusceptibility(props.HasModel):

    chi = props.PhysicalProperty("Magnetic Susceptibility (SI)")

    def __init__(self, chi=None, **kwargs):
        super().__init__(**kwargs)
        self.chi = chi


@props._add_deprecated_physical_property_functions("thicknesses")
class LayerThickness(props.HasModel):
    thicknesses = props.PhysicalProperty("layer thicknesses (m)")

    def __init__(self, thicknesses=None, **kwargs):
        super().__init__(**kwargs)
        if thicknesses is None:
            thicknesses = []
        self.thicknesses = thicknesses


@props._add_deprecated_physical_property_functions("eta")
class ElectricalChargeability(props.HasModel):
    eta = props.PhysicalProperty("Electrical Chargeability (V/V)")

    def __init__(self, eta=None, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta


@props._add_deprecated_physical_property_functions("slowness")
@props._add_deprecated_physical_property_functions("velocity")
class AcousticVelocity(props.HasModel):
    slowness = props.PhysicalProperty("Slowness model (s/m)")
    velocity = props.PhysicalProperty("Velocity model (m/s)")
    slowness.set_reciprocal(velocity)

    def __init__(self, slowness=None, velocity=None, **kwargs):
        super().__init__(**kwargs)
        self.slowness = slowness
        self.velocity = velocity


@props._add_deprecated_physical_property_functions("Ks")
class HydraulicConductivity(props.HasModel):
    Ks = props.PhysicalProperty("Saturated hydraulic conductivity")

    def __init__(self, Ks=None, **kwargs):
        self.Ks = Ks
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


@props._add_deprecated_physical_property_functions("theta_r")
@props._add_deprecated_physical_property_functions("theta_s")
class WaterRetention(props.HasModel):

    theta_r = props.PhysicalProperty("residual water content [L3L-3]")
    theta_s = props.PhysicalProperty("saturated water content [L3L-3]")

    def __init__(
        self,
        theta_r=None,
        theta_s=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta_r = theta_r
        self.theta_s = theta_s

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
