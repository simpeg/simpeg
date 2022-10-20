"""
=======================================================================
Richards Flow (:mod:`SimPEG.flow.richards`)
=======================================================================
.. currentmodule:: SimPEG.flow.richards

About ``Richards flow``

Simulations
===========
.. autosummary::
  :toctree: generated/

  SimulationNDCellCentered

Survey, Sources and Receivers
=============================
.. autosummary::
  :toctree: generated/

  receivers.Pressure
  receivers.Saturation
  survey.Survey

Empirical utilities
===================
.. autosummary::
  :toctree: generated/

  empirical.NonLinearModel
  empirical.BaseWaterRetention
  empirical.BaseHydraulicConductivity
  empirical.Haverkamp_theta
  empirical.Haverkamp_k
  empirical.haverkamp
  empirical.HaverkampParams
  empirical.Vangenuchten_theta
  empirical.Vangenuchten_k
  empirical.van_genuchten
  empirical.VanGenuchtenParams

"""
from . import empirical
from .survey import Survey
from .simulation import SimulationNDCellCentered
from . import receivers

SimulationNDCellCentred = SimulationNDCellCentered
