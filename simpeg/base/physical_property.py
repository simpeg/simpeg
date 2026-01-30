import warnings
import numpy as np
from scipy.constants import mu_0
from simpeg import props


class ElectricalConductivity(props.HasModel):
    """The electrical conductivity property model base class.

    This class is meant to be used when a simulation has electrical conductivity or
    resistivity as a dependent physical property.

    Parameters
    ----------
    sigma, rho : float, array_like, optional
        The electrical conductivity (`sigma`) in S/m (or resistivity, `rho`, in Ohm*m) physical
        property. These two properties have a reciprocal relationship. Both can be accessed, but only one
        can be set at a time. If one is set, the other is cleared.
    sigmaMap, rhoMap : simpeg.maps.IdentityMap, optional
        The mappings for the `sigma` and `rho` physical property. If set the corresponding physical property
        will be calculated using this mapping and the simulation's `model`, and also enable the derivative of that
        physical property with respect to the `model` needed for inversions. The reciprocal property will automatically
        be mapped with a `ReciprocalMap` applied after the set map.
    """

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
    """The magnetic permeability property model base class.

    This class is meant to be used when a simulation has magnetic permeability as a dependent physical property.

    Parameters
    ----------
    mu, mui : float, array_like, optional
        The magnetic permeability (`mu`) in H/m (or inverse magnetic permeability, `mui`, in m/H) physical
        property. These two properties have a reciprocal relationship. Both can be accessed, but only one can be
        set at a time. If one is set, the other is cleared.
    muMap, muiMap : simpeg.maps.IdentityMap, optional
        The mappings for the `mu` and `mui` physical property. If set the corresponding physical property
        will be calculated using this mapping and the simulation's `model`, and also enable the derivative of that
        physical property with respect to the `model` needed for inversions. The reciprocal property will automatically
        be mapped with a `ReciprocalMap` applied after the set map.
    """

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
    """The dielectric permittivity property model base class.

    This class is meant to be used when a simulation has dielectric permittivity as a dependent physical property.

    Parameters
    ----------
    permittivity : float, array_like, optional
        The dielectric permittivity in F/m physical property.

        .. warning::
            For most implemented simulations that use this property have not been thoroughly tested,
            and it does not yet support derivatives. Contributions are welcome!
    """

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
    """The mass density property model base class.

    This class is meant to be used when a simulation has mass density as a dependent physical property.

    Parameters
    ----------
    rho : float, array_like, optional
        The mass density in g/cc.
    rhoMap : simpeg.maps.IdentityMap, optional
        The mapping for the mass density property `rho`. If set, `rho` will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of `rho` with respect to the
        `model` needed for inversions.
    """

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)")

    def __init__(self, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.rhoMap = rhoMap


class MagneticSusceptibility(props.HasModel):
    """The magnetic susceptibility model base class.

    This class is meant to be used when a simulation has magnetic susceptibility as a dependent physical property.

    Parameters
    ----------
    chi : float, array_like, optional
        The magnetic susceptibility in SI units (T/T).
    chiMap : simpeg.maps.IdentityMap, optional
        The mapping for the magnetic susceptibility property `chi`. If set, `chi` will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of `chi` with respect to the
        `model` needed for inversions.
    """

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(self, chi=None, chiMap=None, **kwargs):
        super().__init__(**kwargs)
        self.chi = chi
        self.chiMap = chiMap


class LayerThickness(props.HasModel):
    """The layered thickness property model base class.

    This class is meant to be used when a simulation has a layered thickness dependent physical property.

    Parameters
    ----------
    thicknesses : float, array_like, optional
        The layer thicknesses in meters. Defaults to an empty list.
    thicknessesMap : simpeg.maps.IdentityMap, optional
        The mapping for the `thicknesses` property. If set, `thicknesses` will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of `thicknesses` with respect to the
        `model` needed for inversions.
    """

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
    """The electrical chargeability property model base class.

    This class is meant to be used when a simulation has electrical chargeability as a dependent physical property.

    Parameters
    ----------
    eta : float, array_like, optional
        The electrical chargeability in V/V.
    etaMap : simpeg.maps.IdentityMap, optional
        The mapping for the electrical chargeability `eta` property. If set, `eta` will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of `eta` with respect to the
        `model` needed for inversions.
    """

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    def __init__(self, eta=None, etaMap=None, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.etaMap = etaMap


class AcousticVelocity(props.HasModel):
    """The acoustic velocity property model base class.

    This class is meant to be used when a simulation has acoustic velocity or
    slowness as a dependent physical property.

    Parameters
    ----------
    slowness, velocity : float, array_like, optional
        The `slowness` (in s/m), or `velocity` (in m/s) physical properties. These two properties have a reciprocal
        relationship. Both can be accessed, but only one can be set at a time. If one is set, the other is cleared.
    slownessMap, velocityMap : simpeg.maps.IdentityMap, optional
        The mappings for the `slowness` and `velocity` physical property. If set the corresponding physical property
        will be calculated using this mapping and the simulation's `model`, and also enable the derivative of that
        physical property with respect to the `model` needed for inversions. The reciprocal property will automatically
        be mapped with a `ReciprocalMap` applied after the set map.
    """

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
    """The hydraulic conductivity property model base class.

    This class is meant to be used when a simulation has hydraulic conductivity as a dependent physical property.

    Parameters
    ----------
    Ks : float, array_like, optional
        The saturated hydraulic conductivity.
    KsMap : simpeg.maps.IdentityMap, optional
        The mapping for the hydraulic conductivity `Ks` property. If set, `Ks` will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of `Ks` with respect to the
        `model` needed for inversions.
    """

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
    """The water saturation property model base class.

    This class is meant to be used when a simulation has water saturation (or retention) as a dependent physical property.

    Parameters
    ----------
    theta_r, theta_s : float, array_like, optional
        The residual (`theta_r`) and saturated (`theta_s`) water content physical properties.
    theta_rMap, theta_sMap : simpeg.maps.IdentityMap, optional
        The mapping for the respective water content properties. If set, that physical property will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of the property with respect to the
        `model` needed for inversions.
    """

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


class ColeCole(props.HasModel):
    r"""The cole-cole parameterization model base class.

    This class is meant to be used when a simulation is dependant upon a cole-cole model.

    Parameters
    ----------
    tau, taui : float, array_like, optional
        The Cole-Cole time constant (`tau`) and its inverse (`taui`) parameters.
    c : float, array_like, optional
        The Cole-Cole frequency exponent (`c`) parameter.
    tauMap, tauiMap : simpeg.maps.IdentityMap, optional
        The mapping for the respective Cole-Cole parameters. If set, that parameter will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of the property with respect to the
        `model` needed for inversions.
    cMap : simpeg.maps.IdentityMap, optional
        The mapping for the 'c' Cole-Cole parameter. If set, this parameter will be calculated using
        this mapping and the simulation's `model`, and also enable the derivative of the property needed
        for inversions.

    Notes
    -----

    The Cole-Cole parameterization expresses electrical conductivity :math:`\sigma` as a function of frequency as

    .. math::

        \sigma(\omega) = \sigma_{\infty} - \frac{\eta \sigma_{\infty}}{1 + (i \omega \tau)^{c}}

    with the electrical chargeability, :math:`\eta`, defined as

    .. math::

        \eta = \frac{\sigma_\infty - \sigma_0}{\sigma_\infty}

    See Also
    --------
    ElectricalConductivity, ElectricalChargeability

    """

    tau, tauMap, tauDeriv = props.Invertible("Time constant (s)")
    taui, tauiMap, tauiDeriv = props.Invertible("Inverse of time constant (1/s)")
    props.Reciprocal(tau, taui)

    c, cMap, cDeriv = props.Invertible("Frequency dependency")

    def __init__(
        self,
        tau=None,
        tauMap=None,
        taui=None,
        tauiMap=None,
        c=None,
        cMap=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.tauMap = tauMap
        self.taui = taui
        self.tauiMap = tauiMap
        self.c = c
        self.cMap = cMap


class ViscousMagneticSusceptibility(props.HasModel):
    """The viscous remanent magnetic susceptibility parameterization model base class

    Parameters
    ----------
    dchi : float, array_like, optional
        Frequency dependence parameter
    tau1 : float, array_like, optional
        Lower bound for log-uniform distribution of time-relaxation constants (s)
    tau2 : float, array_like, optional
        Upper bound for log-uniform distribution of time-relaxation constants (s)
    """

    dchi = props.PhysicalProperty("Frequency dependence parameter")
    tau1 = props.PhysicalProperty(
        "Lower bound for log-uniform distribution of time-relaxation constants (s)"
    )
    tau2 = props.PhysicalProperty(
        "Upper bound for log-uniform distribution of time-relaxation constants (s)"
    )

    def __init__(self, dchi=None, tau1=None, tau2=None, **kwargs):
        super().__init__(**kwargs)
        self.dchi = dchi
        self.tau1 = tau1
        self.tau2 = tau2


class AmalgamatedViscousMagneticSusceptibility(props.HasModel):
    r"""The amalgamated viscous remanent magnetic susceptibility parameterization model base class

    Parameters
    ----------
    xi : float, array_like, optional
        Amalgamated Viscous Remanent Magnetization Parameter
    xiMap :  : simpeg.maps.IdentityMap, optional

    Notes
    -----

    .. math::
        \xi = \frac{\Delta\chi}{\log(\tau_2/\tau_1)}

    See Also
    --------
    ViscousMagneticSusceptibility
    """

    xi, xiMap, xiDeriv = props.Invertible(
        "Amalgamated Viscous Remanent Magnetization Parameter"
    )

    def __init__(self, xi=None, xiMap=None, **kwargs):
        super().__init__(**kwargs)
        self.xi = xi
        self.xiMap = xiMap
