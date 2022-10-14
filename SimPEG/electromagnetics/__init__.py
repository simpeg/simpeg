"""
=================================================================
Base EM (:mod:`SimPEG.electromagnetics`)
=================================================================
.. currentmodule:: SimPEG.electromagnetics

About ``electromagnetics``

Base Classes
============
.. autosummary::
  :toctree: generated/

  base.BaseEMSimulation
  base.BaseEMSrc

Analytic Functions
==================
Many of these functions call functions from `geoana`, and we recommend calling
those functions directly from that package.

.. autosummary::
  :toctree: generated/

  analytics.hzAnalyticDipoleT
  analytics.hzAnalyticCentLoopT
  analytics.hzAnalyticDipoleF
  analytics.getCasingEphiMagDipole
  analytics.getCasingHrMagDipole
  analytics.getCasingHzMagDipole
  analytics.getCasingBrMagDipole
  analytics.getCasingBzMagDipole

"""
from scipy.constants import mu_0, epsilon_0

from . import time_domain
from . import frequency_domain
from . import natural_source
from . import analytics
from . import utils
from .static import resistivity, induced_polarization, spectral_induced_polarization
