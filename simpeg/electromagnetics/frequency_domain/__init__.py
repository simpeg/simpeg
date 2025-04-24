r"""
==============================================================================
Frequency-Domain EM (:mod:`simpeg.electromagnetics.frequency_domain`)
==============================================================================
.. currentmodule:: simpeg.electromagnetics.frequency_domain

The ``frequency_domain`` module contains functionality for solving Maxwell's equations
in the frequency-domain for controlled sources. Where a :math:`+i\omega t`
Fourier convention is used, this module is used to solve problems of the form:

.. math::
    \begin{align}
    \nabla \times \vec{E} + i\omega \vec{B} &= - i \omega \vec{S}_m \\
    \nabla \times \vec{H} - \vec{J} &= \vec{S}_e
    \end{align}

where the constitutive relations between fields and fluxes are given by:

* :math:`\vec{J} = (\sigma + i \omega \varepsilon) \vec{E}`
* :math:`\vec{B} = \mu \vec{H}`

and:

* :math:`\vec{S}_m` represents a magnetic source term
* :math:`\vec{S}_e` represents a current source term

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
