from scipy.constants import mu_0

from simpeg import props
from simpeg.simulation import BaseSimulation


class BaseElectricalSimulation(BaseSimulation):
    conductivity, conductivity_map, _con_deriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )
    resistivity, resistivity_map, _res_deriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )
    props.Reciprocal(conductivity, resistivity)

    def __init__(
        self,
        conductivity=None,
        conductivity_map=None,
        resistivity=None,
        resistivity_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conductivity = conductivity
        self.resistivity = resistivity
        self.conductivity_map = conductivity_map
        self.resistivity_map = resistivity_map


class BaseMagneticSimulation(BaseSimulation):
    permeability, permeability_map, _perm_deriv = props.Invertible(
        "Magnetic Permeability (H/m)",
    )
    _perm_inv, _perm_inv_map, _perm_inv_deriv = props.Invertible(
        "Inverse Magnetic Permeability (m/H)"
    )
    props.Reciprocal(permeability, _perm_inv)

    def __init__(self, permeability=mu_0, permeability_map=None, **kwargs):
        super().__init__(**kwargs)
        self.permeability = permeability
        self.permeability_map = permeability_map
        self._perm_inv = None
        self._perm_inv_map = None


class BaseDensitySimulation(BaseSimulation):
    density, density_map, _density_deriv = props.Invertible(
        "Mass density (g/cc)",
    )

    def __init__(self, density=None, density_map=None, **kwargs):
        super().__init__(**kwargs)
        self.density = density
        self.density_map = density_map
