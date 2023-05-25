"""
========================================================
Base SimPEG Classes (:mod:`SimPEG`)
========================================================
.. currentmodule:: SimPEG

SimPEG is built off of several base classes that define the
general structure of simulations and inversion operations.

Simulations
===========

Base Simulations
----------------

.. autosummary::
  :toctree: generated/

  simulation.BaseSimulation
  simulation.BaseTimeSimulation
  simulation.LinearSimulation
  simulation.ExponentialSinusoidSimulation
  base.BasePDESimulation
  base.BaseElectricalPDESimulation
  base.BaseMagneticPDESimulation

Base Surveys, Sources and Receivers
-----------------------------------
.. autosummary::
  :toctree: generated/

  survey.BaseRx
  survey.BaseTimeRx
  survey.BaseSrc
  survey.BaseSurvey
  survey.BaseTimeSurvey

Models
-----------------------------------
.. autosummary::
  :toctree: generated/

  models.Model
  props.PhysicalProperty
  props.Derivative
  props.Invertible
  props.Reciprocal
  props.HasModel

Data
----
.. autosummary::
  :toctree: generated/

  data.Data
  data.SyntheticData

Fields
------
.. autosummary::
  :toctree: generated/

  fields.Fields
  fields.TimeFields

Mappings
--------
.. autosummary::
  :toctree: generated/

  maps.BaseParametric
  maps.ChiMap
  maps.ComboMap
  maps.ComplexMap
  maps.ExpMap
  maps.LinearMap
  maps.IdentityMap
  maps.InjectActiveCells
  maps.MuRelative
  maps.LogMap
  maps.ParametricBlock
  maps.ParametricCircleMap
  maps.ParametricEllipsoid
  maps.ParametricLayer
  maps.ParametricPolyMap
  maps.Projection
  maps.ReciprocalMap
  maps.SphericalSystem
  maps.Surject2Dto3D
  maps.SurjectFull
  maps.SurjectUnits
  maps.SurjectVertical1D
  maps.Weighting
  maps.Wires

Inversions
==========

Objective Function Pieces
-------------------------
.. autosummary::
  :toctree: generated/

  objective_function.BaseObjectiveFunction
  objective_function.ComboObjectiveFunction
  objective_function.L2ObjectiveFunction
  data_misfit.BaseDataMisfit
  data_misfit.L2DataMisfit

Optimizations
-------------
.. autosummary::
  :toctree: generated/

  optimization.ProjectedGradient
  optimization.BFGS
  optimization.GaussNewton
  optimization.InexactGaussNewton
  optimization.SteepestDescent
  optimization.NewtonRoot
  optimization.ProjectedGNCG
  optimization.Minimize
  optimization.Remember
  optimization.IterationPrinters
  optimization.StoppingCriteria

Base inversion pieces
---------------------
.. autosummary::
  :toctree: generated/

  inverse_problem.BaseInvProblem
  inversion.BaseInversion

"""

# import discretize as Mesh
import discretize
from discretize import tests

from .data import Data, SyntheticData
from . import data_misfit
from . import directives
from . import maps
from . import models
from . import inverse_problem
from . import inversion
from . import regularization
from . import survey
from . import simulation

from . import utils
from .utils import mkvc
from .utils import Report
from .utils.solver_utils import (
    _checkAccuracy,
    SolverWrapD,
    SolverWrapI,
    Solver,
    SolverCG,
    SolverDiag,
    SolverLU,
    SolverBiCG,
)

__version__ = "0.19.0"
__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2013 - 2020, SimPEG Team, http://simpeg.xyz"
