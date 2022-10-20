"""
===========================================================================================================
Viscous Remanent Magnetization (:mod:`SimPEG.electromagnetics.viscous_remanent_magnetization`)
===========================================================================================================
.. currentmodule:: SimPEG.electromagnetics.viscous_remanent_magnetization

About ``viscous_remanent_magnetization``

Simulations
===========

.. autosummary::
  :toctree: generated/

  Simulation3DLinear
  Simulation3DLogUniform

Receivers
=========

.. autosummary::
  :toctree: generated/

  receivers.Point
  receivers.SquareLoop


Waveforms
=========

.. autosummary::
  :toctree: generated/

  waveforms.StepOff
  waveforms.SquarePulse
  waveforms.ArbitraryDiscrete
  waveforms.ArbitraryPiecewise
  waveforms.Custom
  waveforms.BaseVRMWaveform


Sources
=======

.. autosummary::
  :toctree: generated/

  sources.MagDipole
  sources.CircLoop
  sources.LineCurrent

Surveys
=======
.. autosummary::
  :toctree: generated/

  Survey

Base Classes
============
.. autosummary::
  :toctree: generated/

  BaseVRMSimulation
  sources.BaseSrcVRM
  waveforms.BaseVRMWaveform

"""
from . import receivers
from . import sources
from . import receivers as Rx
from . import sources as Src
from . import waveforms

from .simulation import BaseVRMSimulation, Simulation3DLinear, Simulation3DLogUniform

from .survey import SurveyVRM as Survey
