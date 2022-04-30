"""
==============================================================================
Natural Source EM Module (:mod:`SimPEG.electromagnetics.natural_source`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.natural_source

About ``natural_source``

Receiver Classes
----------------

.. autosummary::
  :toctree: generated/

  receivers.PointNaturalSource
  receivers.Point3DTipper

Source Classes
--------------
.. autosummary::
  :toctree: generated/

  sources.Planewave
  sources.PlanewaveXYPrimary

Survey Classes
--------------
.. autosummary::
  :toctree: generated/

  survey.Data

"""

from . import utils
from . import sources as Src
from . import receivers as Rx
from .survey import Survey, Data
from .fields import Fields1DPrimarySecondary, Fields3DPrimarySecondary
from .simulation import Simulation1DPrimarySecondary, Simulation3DPrimarySecondary
from . import sources
from . import receivers
from .simulation_1d import Simulation1DRecursive
