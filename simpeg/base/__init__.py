from .physical_property import (
    ElectricalConductivity,
    MagneticPermeability,
    DielectricPermittivity,
    MassDensity,
    LayerThickness,
    ElectricalChargeability,
    MagneticSusceptibility,
    AcousticVelocity,
    HydraulicConductivity,
    WaterRetention,
    ColeCole,
    ViscousMagneticSusceptibility,
    AmalgamatedViscousMagneticSusceptibility,
)

from .pde_simulation import (
    BasePDESimulation,
    BaseElectricalPDESimulation,
    BaseMagneticPDESimulation,
    with_property_mass_matrices,
)
