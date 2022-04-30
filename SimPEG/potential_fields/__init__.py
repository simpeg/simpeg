"""
=================================================================
Base Classes and Functions (:mod:`SimPEG.potential_fields`)
=================================================================
.. currentmodule:: SimPEG.potential_fields

About ``potential_fields``

.. autosummary::
  :toctree: generated/

  base.BasePFSimulation
  base.progress
  base.get_dist_wgt


"""
from __future__ import absolute_import

from . import magnetics
from . import gravity

from .base import get_dist_wgt
