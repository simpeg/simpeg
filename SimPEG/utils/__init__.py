from __future__ import print_function

from discretize.utils.interputils import interpmat

from .matutils import (
    mkvc, sdiag, sdInv, speye, kron3, spzeros, ddx, av,
    av_extrap, ndgrid, ind2sub, sub2ind, getSubArray,
    inv3X3BlockDiagonal, inv2X2BlockDiagonal, TensorType,
    makePropertyTensor, invPropertyTensor, diagEst, Zero,
    Identity, uniqueRows, cartesian2spherical, spherical2cartesian,
    coterminal
)
from .codeutils import (
    memProfileWrapper, hook, setKwargs,
    printTitles, printLine, checkStoppers, printStoppers, printDone,
    callHooks, dependentProperty,
    asArray_N_x_Dim, requires,
    Report
)
from .meshutils import (
    exampleLrmGrid, meshTensor, closestPoints, ExtractCoreMesh
)
from .curvutils import volTetra, faceInfo, indexCube
from .counterutils import Counter, count, timeIt
from . import modelbuilder
from . import solverutils
from .coordutils import rotatePointsFromNormals, rotationMatrixFromNormals
from .modelutils import surface2ind_topo
from .PlotUtils import plot2Ddata, plotLayer
from .io_utils import download

"""
Deprecated
"""

SolverUtils = solverutils
ModelBuilder = modelbuilder
import .counterutils as CounterUtils
