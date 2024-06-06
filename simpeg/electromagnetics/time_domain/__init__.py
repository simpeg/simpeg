r"""
==============================================================================
Time-Domain EM (:mod:`simpeg.electromagnetics.time_domain`)
==============================================================================
.. currentmodule:: simpeg.electromagnetics.time_domain

The ``time_domain`` module contains functionality for solving Maxwell's equations
in the time-domain for controlled sources. Here, electric displacement is ignored,
and functionality is used to solve:

.. math::
    \begin{align}
    \nabla \times \vec{e} + \frac{\partial \vec{b}}{\partial t} &= -\frac{\partial \vec{s}_m}{\partial t} \\
    \nabla \times \vec{h} - \vec{j} &= \vec{s}_e
    \end{align}

where the constitutive relations between fields and fluxes are given by:

* :math:`\vec{j} = \sigma \vec{e}`
* :math:`\vec{b} = \mu \vec{h}`

and:

* :math:`\vec{s}_m` represents a magnetic source term
* :math:`\vec{s}_e` represents a current source term

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
