"""
==============================================================================
Frequency-Domain EM Module (:mod:`SimPEG.electromagnetics.frequency_domain`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.frequency_domain

About ``frequency_domain``

Receiver Classes
----------------

.. autosummary::
  :toctree: generated/

  receivers.BaseRx
  receivers.PointElectricField
  receivers.PointMagneticFluxDensity
  receivers.PointMagneticFluxDensitySecondary
  receivers.PointMagneticField
  receivers.PointCurrentDensity

Source Classes
--------------
.. autosummary::
  :toctree: generated/

  sources.BaseFDEMSrc
  sources.RawVec_e
  sources.RawVec_m
  sources.RawVec
  sources.MagDipole
  sources.MagDipole_Bfield
  sources.CircularLoop
  sources.PrimSecSigma
  sources.PrimSecMappedSigma
  sources.LineCurrent

Survey Classes
--------------
.. autosummary::
  :toctree: generated/

  survey.Survey

"""
from .survey import Survey
from . import sources
from . import receivers
from .simulation import (
    Simulation3DElectricField,
    Simulation3DMagneticFluxDensity,
    Simulation3DCurrentDensity,
    Simulation3DMagneticField,
)
from .fields import (
    Fields3DElectricField,
    Fields3DMagneticFluxDensity,
    Fields3DCurrentDensity,
    Fields3DMagneticField,
)

from . import sources as Src
from . import receivers as Rx
