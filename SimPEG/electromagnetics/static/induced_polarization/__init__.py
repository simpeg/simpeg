"""
============================================================================================
Induced Polarization (:mod:`SimPEG.electromagnetics.static.induced_polarization`)
============================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.induced_polarization


Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation2DCellCentered
  Simulation2DNodal
  Simulation3DCellCentered
  Simulation3DNodal

Receivers, Sources, and Surveys
===============================
The ``induced_polarization`` module makes use of receivers, sources, and surveys
defined in the ``SimPEG.electromagnetics.static.resistivity`` module.
"""
from .simulation import (
    Simulation3DCellCentered,
    Simulation3DNodal,
    Simulation2DCellCentered,
    Simulation2DNodal,
)
from .survey import from_dc_to_ip_survey
from .run import run_inversion
from ..resistivity.survey import Survey
from ..resistivity import receivers
from ..resistivity import sources
from ..resistivity import utils

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered
