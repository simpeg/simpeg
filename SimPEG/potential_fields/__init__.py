"""
=================================================================
Base Classes and Functions (:mod:`SimPEG.potential_fields`)
=================================================================
.. currentmodule:: SimPEG.potential_fields

About ``potential_fields``

Base classes and utility functions
==================================

.. autosummary::
  :toctree: generated/

  base.BasePFSimulation
  base.BaseEquivalentSourceLayerSimulation
  base.progress
  get_dist_wgt


"""

from . import magnetics
from . import gravity

from .base import get_dist_wgt
