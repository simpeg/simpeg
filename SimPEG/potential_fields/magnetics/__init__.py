"""
=========================================================================
Magnetics Simulation Module (:mod:`SimPEG.potential_fields.magnetics`)
=========================================================================
.. currentmodule:: SimPEG.potential_fields.magnetics

About ``magnetics``

Survey, Source and Receiver Classes
-----------------------------------

.. autosummary::
  :toctree: generated/

  receivers.Point
  sources.SourceField
  survey.Survey


"""
from . import survey
from . import sources
from . import receivers
from . import analytics
from . import simulation

from .simulation import (
    Simulation3DIntegral,
    SimulationEquivalentSourceLayer,
    Simulation3DDifferential,
)
from .survey import Survey
from .sources import SourceField
from .receivers import Point
