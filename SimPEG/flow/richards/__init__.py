"""
=======================================================================
Richards Flow Module (:mod:`SimPEG.flow.richards`)
=======================================================================
.. currentmodule:: SimPEG.flow.richards

About ``gravity``

Simulations
-----------
.. autosummary::
  :toctree: generated/
  SimulationNDCellCentered

Survey, Source and Receiver Classes
-----------------------------------
.. autosummary::
  :toctree: generated/

  receivers.Pressure
  receivers.Saturation
  survey.Survey

Empirical utilities
-------------------
.. autosummary::
  :toctree: generated/

  empirical.Pressure
  empirical.Saturation
  empirical.Survey

"""
from . import empirical
from .survey import Survey
from .simulation import SimulationNDCellCentered
from . import receivers

SimulationNDCellCentred = SimulationNDCellCentered
