"""
=================================================================
Base Classes (:mod:`SimPEG.electromagnetics`)
=================================================================
.. currentmodule:: SimPEG.electromagnetics

About ``electromagnetics``

.. autosummary::
  :toctree: generated/

  base.BaseEMSrc

"""
from scipy.constants import mu_0, epsilon_0

from . import time_domain
from . import frequency_domain
from . import natural_source
from . import analytics
from . import utils
from .static import resistivity, induced_polarization, spectral_induced_polarization
