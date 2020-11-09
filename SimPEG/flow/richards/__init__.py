# Copyright (c) 2013 SimPEG Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
# This code is part of the SimPEG project (https://simpeg.xyz)
from . import empirical
from .survey import Survey
from .simulation import SimulationNDCellCentered
from . import receivers

SimulationNDCellCentred = SimulationNDCellCentered

############
# Deprecated
############
from .simulation import RichardsProblem
