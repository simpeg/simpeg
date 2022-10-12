"""
===========================================================================================================
Viscous Remanent Magnetization Module (:mod:`SimPEG.electromagnetics.viscous_remanent_magnetization`)
===========================================================================================================
.. currentmodule:: SimPEG.electromagnetics.viscous_remanent_magnetization

About ``viscous_remanent_magnetization``

Receiver Classes
----------------

.. autosummary::
  :toctree: generated/

  receivers.Point
  receivers.SquareLoop


Waveform Functions and Classes
------------------------------

.. autosummary::
  :toctree: generated/

  waveforms.StepOff
  waveforms.SquarePulse
  waveforms.ArbitraryDiscrete
  waveforms.ArbitraryPiecewise
  waveforms.Custom
  waveforms.BaseVRMWaveform


Source Classes
--------------

.. autosummary::
  :toctree: generated/

  sources.BaseSrcVRM
  sources.MagDipole
  sources.CircLoop
  sources.LineCurrent

Survey Classes
--------------
.. autosummary::
  :toctree: generated/

  survey.SurveyVRM

"""
from . import receivers
from . import sources
from . import receivers as Rx
from . import sources as Src
from . import waveforms

from .simulation import BaseVRMSimulation, Simulation3DLinear, Simulation3DLogUniform

from .survey import SurveyVRM as Survey
