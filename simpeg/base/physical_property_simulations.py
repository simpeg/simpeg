from scipy.constants import mu_0

from simpeg import props


class BaseConductivity(props.HasModel):
    sigma, sigmaMap, sigmaDeriv = props.Invertible("Electrical conductivity (S/m)")
    rho, rhoMap, rhoDeriv = props.Invertible("Electrical resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    def __init__(self, sigma=None, sigmaMap=None, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho
        self.sigmaMap = sigmaMap
        self.rhoMap = rhoMap


class BasePermeability(props.HasModel):
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


class BaseDensity(props.HasModel):

    rho, rhoMap, rhoDeriv = props.Invertible("Specific density (g/cc)")

    def __init__(self, rho=None, rhoMap=None, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho
        self.rhoMap = rhoMap


class BaseSusceptibility(props.HasModel):

    chi, chiMap, chiDeriv = props.Invertible("Magnetic Susceptibility (SI)")

    def __init__(self, chi=None, chiMap=None, **kwargs):
        super().__init__(**kwargs)
        self.chi = chi
        self.chiMap = chiMap


class BaseThickness(props.HasModel):
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "layer thicknesses (m)"
    )

    def __init__(self, thicknesses=None, thicknessesMap=None, **kwargs):
        super().__init__(**kwargs)
        if thicknesses is None:
            thicknesses = []
        self.thicknesses = thicknesses
        self.thicknessesMap = thicknessesMap


class BaseChargeability(props.HasModel):
    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    def __init__(self, eta=None, etaMap=None, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.etaMap = etaMap


class BaseVelocity(props.HasModel):
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
