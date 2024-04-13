"""
============================================================================================
Self Potential (:mod:`SimPEG.electromagnetics.static.self_potential`)
============================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.self_potential


Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation3DCellCentered

Receivers
=========
This module makes use of the receivers in :mod:`SimPEG.electromagnetics.static.resistivity`

Sources
=======
.. autosummary::
  :toctree: generated/

  sources.StreamingCurrents

Surveys
=======
.. autosummary::
  :toctree: generated/

  Survey

Maps
====
The self potential simulation provides two specialized maps to extend to inversions
with different types of model sources.

.. autosummary::
  :toctree: generated/

  CurrentDensityMap
  HydraulicHeadMap

"""

from .simulation import (
    Simulation3DCellCentered,
    Survey,
    CurrentDensityMap,
    HydraulicHeadMap,
)
from . import sources
