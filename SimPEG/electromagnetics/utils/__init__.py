"""
===================================================================
Electromagnetics Utilities (:mod:`SimPEG.electromagnetics.utils`)
===================================================================

.. currentmodule:: SimPEG.electromagnetics.utils


Current Utilities
=================

.. autosummary::
  :toctree: generated/

  edge_basis_function
  getStraightLineCurrentIntegral
  segmented_line_current_source_term
  line_through_faces
  getSourceTermLineCurrentPolygon

Waveform Utilities
==================

.. autosummary::
  :toctree: generated/

  omega
  k
  VTEMFun
  convolve_with_waveform

"""
from .waveform_utils import (
    omega,
    k,
    VTEMFun,
    convolve_with_waveform,
)

from .current_utils import (
    edge_basis_function,
    getStraightLineCurrentIntegral,
    getSourceTermLineCurrentPolygon,
    segmented_line_current_source_term,
    line_through_faces,
)
