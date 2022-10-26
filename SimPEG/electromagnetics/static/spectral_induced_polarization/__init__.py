"""
====================================================================================================
Spectral Induced Polarization (:mod:`SimPEG.electromagnetics.static.induced_polarization`)
====================================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.spectral_induced_polarization


Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation2DCellCentered
  Simulation2DNodal
  Simulation3DCellCentered
  Simulation3DNodal


Receivers
=========

.. autosummary::
  :toctree: generated/

  receivers.Dipole
  receivers.Pole

Sources
=======
.. autosummary::
  :toctree: generated/

  sources.Dipole
  sources.Pole

Surveys
=======
.. autosummary::
  :toctree: generated/

  survey.Survey

Utilities
=========
.. autosummary::
  :toctree: generated/

  run_inversion
  from_dc_to_sip_survey
  spectral_ip_mappings

Base Classes
============
.. autosummary::
  :toctree: generated/

  receivers.BaseRx
  sources.BaseSrc
  simulation.BaseSIPSimulation
  simulation_2d.BaseSIPSimulation2D

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
