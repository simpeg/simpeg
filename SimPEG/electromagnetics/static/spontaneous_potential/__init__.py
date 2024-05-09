"""
============================================================================================
Spontaneous Potential (:mod:`simpeg.electromagnetics.static.spontaneous_potential`)
============================================================================================
.. currentmodule:: simpeg.electromagnetics.static.spontaneous_potential

.. admonition:: important

  This module will be deprecated in favour of ``simpeg.electromagnetics.static.self_potential``


Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation3DCellCentered

Receivers
=========
This module makes use of the receivers in :mod:`simpeg.electromagnetics.static.resistivity`

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
The spontaneous potential simulation provides two specialized maps to extend to inversions
with different types of model sources.

.. autosummary::
  :toctree: generated/

  CurrentDensityMap
  HydraulicHeadMap

"""

import warnings

warnings.warn(
    (
        "The 'spontaneous_potential' module has been renamed to 'self_potential'. "
        "Please use the 'self_potential' module instead. "
        "The 'spontaneous_potential' module will be removed in SimPEG 0.23."
    ),
    FutureWarning,
    stacklevel=2,
)

from ..self_potential.simulation import (
    Simulation3DCellCentered,
    Survey,
    CurrentDensityMap,
    HydraulicHeadMap,
)
from ..self_potential import sources
from ..self_potential import simulation
