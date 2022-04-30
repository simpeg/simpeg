"""
====================================================================================================
Spectral Induced Polarization Module (:mod:`SimPEG.electromagnetics.static.induced_polarization`)
====================================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.spectral_induced_polarization



Receiver Classes
----------------

.. autosummary::
  :toctree: generated/

  receivers.BaseRx
  receivers.Dipole
  receivers.Pole

Source Classes
--------------
.. autosummary::
  :toctree: generated/

  sources.BaseSrc
  sources.Dipole
  sources.Pole

Survey Classes
--------------
.. autosummary::
  :toctree: generated/

  survey.Survey

"""
from ....data import Data
from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .simulation_2d import Simulation2DCellCentered, Simulation2DNodal
from .survey import Survey, from_dc_to_sip_survey
from . import sources
from . import receivers
from . import sources as Src
from . import receivers as Rx
from .run import run_inversion, spectral_ip_mappings


Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered
