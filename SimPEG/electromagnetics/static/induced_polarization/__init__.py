"""
============================================================================================
Induced Polarization Module (:mod:`SimPEG.electromagnetics.static.induced_polarization`)
============================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.induced_polarization


Receiver Classes
----------------

Receivers used by the ``induced_polarization`` module are found in the
``SimPEG.electromagnetics.static.resistivity.receivers``.

Source Classes
--------------

Sources used by the ``induced_polarization`` module are found in the
``SimPEG.electromagnetics.static.resistivity.sources``.

Survey Classes
--------------

The survey class used by the ``induced_polarization`` module is found in the
``SimPEG.electromagnetics.static.resistivity.survey``. Below is additional
functionality.

.. autosummary::
  :toctree: generated/

  survey.from_dc_to_ip_survey

"""
from .simulation import (
    Simulation3DCellCentered,
    Simulation3DNodal,
    Simulation2DCellCentered,
    Simulation2DNodal,
)
from .survey import Survey, from_dc_to_ip_survey
from .run import run_inversion
from ..resistivity import receivers
from ..resistivity import sources
from ..resistivity import utils

Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation2DCellCentered
