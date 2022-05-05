"""
===================================================================
Electromagnetics Utilities (:mod:`SimPEG.electromagnetics.utils`)
===================================================================

.. currentmodule:: SimPEG.electromagnetics.utils


Current Utilities
-----------------

.. autosummary::
  :toctree: generated/

  current_utils.edge_basis_function
  current_utils.getStraightLineCurrentIntegral
  current_utils.findlast
  current_utils.segmented_line_current_source_term
  current_utils.line_through_faces
  current_utils.getSourceTermLineCurrentPolygon


Waveform Utilities
------------------

.. autosummary::
  :toctree: generated/

  waveform_utils.omega
  waveform_utils.k
  waveform_utils.TriangleFun
  waveform_utils.TriangleFunDeriv
  waveform_utils.SineFun
  waveform_utils.SineFunDeriv
  waveform_utils.VTEMFun

"""
from .waveform_utils import omega, k, VTEMFun, TriangleFun, SineFun
from .current_utils import (
    edge_basis_function,
    findlast,
    getStraightLineCurrentIntegral,
    getSourceTermLineCurrentPolygon,
    segmented_line_current_source_term,
    line_through_faces,
)
