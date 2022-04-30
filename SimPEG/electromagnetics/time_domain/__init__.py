"""
==============================================================================
Time-Domain EM Module (:mod:`SimPEG.electromagnetics.time_domain`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.time_domain

About ``time_domain``

Receiver Classes
----------------

.. autosummary::
  :toctree: generated/

  receivers.BaseRx
  receivers.PointElectricField
  receivers.PointCurrentDensity
  receivers.PointMagneticFluxDensity
  receivers.PointMagneticFluxTimeDerivative
  receivers.PointMagneticField
  receivers.PointMagneticFieldTimeDerivative


Waveform Functions and Classes
------------------------------

.. autosummary::
  :toctree: generated/

  sources.BaseWaveform
  sources.StepOffWaveform
  sources.RampOffWaveform
  sources.RawWaveform
  sources.VTEMWaveform
  sources.TrapezoidWaveform
  sources.TriangularWaveform
  sources.QuarterSineRampOnWaveform
  sources.HalfSineWaveform


Source Classes
--------------

.. autosummary::
  :toctree: generated/

  sources.BaseTDEMSrc
  sources.MagDipole
  sources.CircularLoop
  sources.LineCurrent
  sources.RawVec_Grounded

Survey Classes
--------------
.. autosummary::
  :toctree: generated/

  survey.Survey

"""
from .simulation import (
    Simulation3DMagneticFluxDensity,
    Simulation3DElectricField,
    Simulation3DMagneticField,
    Simulation3DCurrentDensity,
)
from .fields import (
    Fields3DMagneticFluxDensity,
    Fields3DElectricField,
    Fields3DMagneticField,
    Fields3DCurrentDensity,
)
from .survey import Survey
from . import sources
from . import receivers

from . import sources as Src
from . import receivers as Rx
