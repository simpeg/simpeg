"""
============================================================================================
DC Resistivity (:mod:`SimPEG.electromagnetics.static.resistivity`)
============================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.resistivity


Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation1DLayers
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
  sources.Multipole

Surveys
=======
.. autosummary::
  :toctree: generated/

  Survey

Fields
======
.. autosummary::
  :toctree: generated/

  Fields2DCellCentered
  Fields2DNodal
  Fields3DCellCentered
  Fields3DNodal

Utilities
=========
.. autosummary::
  :toctree: generated/

  IO
  run_inversion
  utils.WennerSrcList

Base Classes
============
.. autosummary::
  :toctree: generated/

  FieldsDC
  Fields2D
  simulation.BaseDCSimulation
  simulation_2d.BaseDCSimulation2D
  sources.BaseSrc
  receivers.BaseRx
"""
from .simulation import Simulation3DCellCentered, Simulation3DNodal
from .simulation_2d import Simulation2DCellCentered, Simulation2DNodal
from .simulation_1d import Simulation1DLayers
from .survey import Survey
from . import sources
from . import receivers
from .fields import FieldsDC, Fields3DCellCentered, Fields3DNodal
from .fields_2d import Fields2D, Fields2DCellCentered, Fields2DNodal
from . import utils
from .IODC import IO
from .run import run_inversion

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation3DCellCentered
Fields3DCellCentred = Fields3DCellCentered
Fields2DCellCentred = Fields2DCellCentered

# Deprecate this eventually....
from . import sources as Src
from . import receivers as Rx
