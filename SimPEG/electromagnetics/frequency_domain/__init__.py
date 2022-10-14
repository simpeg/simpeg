"""
==============================================================================
Frequency-Domain EM (:mod:`SimPEG.electromagnetics.frequency_domain`)
==============================================================================
.. currentmodule:: SimPEG.electromagnetics.frequency_domain

About ``frequency_domain``

Simulations
===========
.. autosummary::
  :toctree: generated/

  Simulation1DLayered
  Simulation3DElectricField
  Simulation3DMagneticFluxDensity
  Simulation3DCurrentDensity
  Simulation3DMagneticField


Receivers
=========
.. autosummary::
  :toctree: generated/

  receivers.PointElectricField
  receivers.PointMagneticFluxDensity
  receivers.PointMagneticFluxDensitySecondary
  receivers.PointMagneticField
  receivers.PointCurrentDensity

Sources
=======
.. autosummary::
  :toctree: generated/

  sources.RawVec_e
  sources.RawVec_m
  sources.RawVec
  sources.MagDipole
  sources.MagDipole_Bfield
  sources.CircularLoop
  sources.PrimSecSigma
  sources.PrimSecMappedSigma
  sources.LineCurrent

Surveys
=======
.. autosummary::
  :toctree: generated/

  survey.Survey

Fields
======
.. autosummary::
  :toctree: generated/

  Fields3DElectricField
  Fields3DMagneticFluxDensity
  Fields3DCurrentDensity
  Fields3DMagneticField

Base Classes
============
.. autosummary::
  :toctree: generated/

  survey.Survey
  sources.BaseFDEMSrc
  receivers.BaseRx
  simulation.BaseFDEMSimulation
  fields.FieldsFDEM

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
from .simulation_1d import Simulation1DLayered
from .fields import (
    Fields3DElectricField,
    Fields3DMagneticFluxDensity,
    Fields3DCurrentDensity,
    Fields3DMagneticField,
)

from . import sources as Src
from . import receivers as Rx
