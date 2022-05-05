"""
==================================================================================
Straight Ray Tomography Module (:mod:`SimPEG.seismic.straight_ray_tomography`)
==================================================================================
.. currentmodule:: SimPEG.seismic.straight_ray_tomography

About ``straight_ray_tomography``

Survey, Source and Receiver Classes
-----------------------------------

.. autosummary::
  :toctree: generated/

  survey.StraightRaySurvey


"""
from .simulation import Simulation2DIntegral as Simulation
from .simulation import lengthInCell
from .survey import StraightRaySurvey as Survey
from ...survey import BaseSrc as Src
from ...survey import BaseRx as Rx
