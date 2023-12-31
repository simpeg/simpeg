from ..base_1d_stitched import BaseStitchedEM1DSimulation
from .simulation_1d import Simulation1DLayered
from .survey import Survey


class Simulation1DLayeredStitched(BaseStitchedEM1DSimulation):
    """
    Stitched 1D simulation for frequency-domain EM.
    """

    _simulation_type = Simulation1DLayered
    _survey_type = Survey
