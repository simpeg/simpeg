from .physical_property_simulations import (
    BaseConductivity,
    BasePermeability,
    BaseDensity,
    BaseThickness,
    BaseChargeability,
    BaseSusceptibility,
    BaseVelocity,
)

from .pde_simulation import (
    BasePDESimulation,
    BaseElectricalPDESimulation,
    BaseMagneticPDESimulation,
    with_property_mass_matrices,
)
