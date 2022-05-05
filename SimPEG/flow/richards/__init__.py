"""
=======================================================================
Richards Flow Module (:mod:`SimPEG.flow.richards`)
=======================================================================
.. currentmodule:: SimPEG.flow.richards

About ``gravity``

Survey, Source and Receiver Classes
-----------------------------------

.. autosummary::
  :toctree: generated/

  receivers.Pressure
  receivers.Saturation
  survey.Survey


"""
from . import empirical
from .survey import Survey
from .simulation import SimulationNDCellCentered
from . import receivers

SimulationNDCellCentred = SimulationNDCellCentered
