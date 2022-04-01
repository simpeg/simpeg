from .code_utils import deprecate_module

deprecate_module("simulation_2d", "simulation", "0.18.0", error=True)

from .simulation import (
    Simulation2DNodal,
    Simulation2DCellCentered,
    Simulation2DCellCentred,
    Problem2D_N,
    Problem3D_N,
)
