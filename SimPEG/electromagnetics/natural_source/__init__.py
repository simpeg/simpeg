"""
==============================================================================
Natural Source EM (:mod:`SimPEG.electromagnetics.natural_source`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.natural_source

About ``natural_source``

Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation1DElectricField
  Simulation1DMagneticField
  Simulation1DPrimarySecondary
  Simulation2DElectricField
  Simulation2DMagneticField
  Simulation3DPrimarySecondary

Receivers
=========
.. autosummary::
  :toctree: generated/

  receivers.PointNaturalSource
  receivers.Point3DTipper

Sources
=======
.. autosummary::
  :toctree: generated/

  sources.Planewave
  sources.PlanewaveXYPrimary

Data
====
.. autosummary::
  :toctree: generated/

  survey.Data

Fields
======
.. autosummary::
  :toctree: generated/

  Fields1DElectricField
  Fields1DMagneticField
  Fields1DPrimarySecondary
  Fields2DElectricField
  Fields2DMagneticField
  Fields3DPrimarySecondary

"""

from . import utils
from . import sources as Src
from . import receivers as Rx
from .survey import Survey, Data
from .fields import (
    Fields1DElectricField,
    Fields1DMagneticField,
    Fields1DPrimarySecondary,
    Fields2DElectricField,
    Fields2DMagneticField,
    Fields3DPrimarySecondary,
)
from .simulation import (
    Simulation1DElectricField,
    Simulation1DMagneticField,
    Simulation1DPrimarySecondary,
    Simulation2DElectricField,
    Simulation2DMagneticField,
    Simulation3DPrimarySecondary,
)
from . import sources
from . import receivers
from .simulation_1d import Simulation1DRecursive
