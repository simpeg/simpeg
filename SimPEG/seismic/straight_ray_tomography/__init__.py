"""
==================================================================================
Straight Ray Tomography (:mod:`simpeg.seismic.straight_ray_tomography`)
==================================================================================
.. currentmodule:: simpeg.seismic.straight_ray_tomography

About ``straight_ray_tomography``

Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation


Survey, Sources and Receivers
=============================

.. autosummary::
  :toctree: generated/

  survey.StraightRaySurvey


"""

from .simulation import Simulation2DIntegral as Simulation
from .survey import StraightRaySurvey as Survey
from ...survey import BaseSrc as Src
from ...survey import BaseRx as Rx
