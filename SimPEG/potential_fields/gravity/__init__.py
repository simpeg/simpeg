"""
=======================================================================
Gravity Simulation Module (:mod:`SimPEG.potential_fields.gravity`)
=======================================================================
.. currentmodule:: SimPEG.potential_fields.gravity

About ``gravity``

Simulations
-----------
Simulation3DIntegral
SimulationEquivalentSourceLayer
Simulation3DDifferential

Survey, Source and Receiver Classes
-----------------------------------

.. autosummary::
  :toctree: generated/

  Point
  SourceField
  Survey


Analytic functions
------------------
.. autosummary::
  :toctree: generated/

  analytics.GravSphereFreeSpace
  analytics.GravityGradientSphereFreeSpace

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
