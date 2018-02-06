from __future__ import print_function

import numpy as np
import discretize
import properties

from . import Utils
from . import Models
from . import Maps
from . import Props


class BaseSimulation(Props.HasModel):
    """
    BaseSimulation is the base class for all geophysical forward simulations in
    SimPEG.
    """

    counter = properties.Instance(
        "A SimPEG.Utils.Counter object",
        Utils.Counter
    )

    # TODO: Solver code needs to be cleaned up so this is either a pymatsolver
    # solver or a SimPEG solver (or similar)
    solver = properties.Instance(
        "Solver for the forward simulation",
        object,
        default=Utils.SolverUtils.Solver
    )

    solver_opts = properties.Instance(
        "solver options as a kwarg dict",
        dict
    )

    mesh = properties.Instance(
        "a discretize mesh instance",
        discretize.BaseMesh
    )


