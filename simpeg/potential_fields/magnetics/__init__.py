"""
=========================================================================
Magnetics Simulation (:mod:`simpeg.potential_fields.magnetics`)
=========================================================================
.. currentmodule:: simpeg.potential_fields.magnetics

About ``magnetics``

Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation3DIntegral
  SimulationEquivalentSourceLayer
  Simulation3DDifferential

Survey, Source and Receiver Classes
===================================

.. autosummary::
  :toctree: generated/

  Point
  UniformBackgroundField
  Survey

Analytics
=========
.. autosummary::
  :toctree: generated/

  analytics.IDTtoxyz
  analytics.MagSphereAnaFun
  analytics.MagSphereAnaFunA
  analytics.MagSphereFreeSpace
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
from .sources import SourceField, UniformBackgroundField
from .receivers import Point
