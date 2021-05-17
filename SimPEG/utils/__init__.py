from __future__ import print_function

from discretize.utils.interputils import interpmat

from .mat_utils import (
    mkvc,
    sdiag,
    sdInv,
    speye,
    kron3,
    spzeros,
    ddx,
    av,
    av_extrap,
    ndgrid,
    ind2sub,
    sub2ind,
    getSubArray,
    inv3X3BlockDiagonal,
    inv2X2BlockDiagonal,
    TensorType,
    makePropertyTensor,
    invPropertyTensor,
    diagEst,
    Zero,
    Identity,
    uniqueRows,
    cartesian2spherical,
    spherical2cartesian,
    coterminal,
)
from .code_utils import (
    memProfileWrapper,
    hook,
    setKwargs,
    printTitles,
    printLine,
    checkStoppers,
    printStoppers,
    printDone,
    callHooks,
    dependentProperty,
    asArray_N_x_Dim,
    requires,
    Report,
)
from .mesh_utils import exampleLrmGrid, meshTensor, closestPoints, ExtractCoreMesh
from .curv_utils import volTetra, faceInfo, indexCube
from .counter_utils import Counter, count, timeIt
from . import model_builder
from . import solver_utils
from .coord_utils import rotatePointsFromNormals, rotationMatrixFromNormals
from .model_utils import surface2ind_topo
from .plot_utils import plot2Ddata, plotLayer
from .io_utils import download

"""
Deprecated,
don't think we can throw warning if a user accesses them from here...
"""
SolverUtils = solver_utils
ModelBuilder = model_builder
