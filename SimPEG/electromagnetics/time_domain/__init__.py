"""
==============================================================================
Time-Domain EM (:mod:`SimPEG.electromagnetics.time_domain`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.time_domain

About ``time_domain``

Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation1DLayered
  Simulation3DMagneticFluxDensity
  Simulation3DElectricField
  Simulation3DMagneticField
  Simulation3DCurrentDensity

Receivers
=========

.. autosummary::
  :toctree: generated/

  receivers.PointElectricField
  receivers.PointCurrentDensity
  receivers.PointMagneticFluxDensity
  receivers.PointMagneticFluxTimeDerivative
  receivers.PointMagneticField
  receivers.PointMagneticFieldTimeDerivative


Waveforms
=========

.. autosummary::
  :toctree: generated/

  sources.StepOffWaveform
  sources.RampOffWaveform
  sources.RawWaveform
  sources.VTEMWaveform
  sources.TrapezoidWaveform
  sources.TriangularWaveform
  sources.QuarterSineRampOnWaveform
  sources.HalfSineWaveform

Sources
=======

.. autosummary::
  :toctree: generated/

  sources.MagDipole
  sources.CircularLoop
  sources.LineCurrent
  sources.RawVec_Grounded

Surveys
=======
.. autosummary::
  :toctree: generated/

  survey.Survey

Fields
======
.. autosummary::
  :toctree: generated/

  Fields3DMagneticFluxDensity
  Fields3DElectricField
  Fields3DMagneticField
  Fields3DCurrentDensity

Base Classes
============

.. autosummary::
  :toctree: generated/

  receivers.BaseRx
  sources.BaseWaveform
  sources.BaseTDEMSrc
  simulation.BaseTDEMSimulation
  fields.FieldsTDEM
  fields.FieldsDerivativesEB
  fields.FieldsDerivativesHJ

"""
from .simulation import (
    Simulation3DMagneticFluxDensity,
    Simulation3DElectricField,
    Simulation3DMagneticField,
    Simulation3DCurrentDensity,
)
from .simulation_1d import Simulation1DLayered
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
