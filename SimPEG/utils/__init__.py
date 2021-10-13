from __future__ import print_function

from discretize.utils.interpolation_utils import interpmat

from .mat_utils import (
    mkvc,
    sdiag,
    sdinv,
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
    get_subarray,
    getSubArray,
    inverse_3x3_block_diagonal,
    inv3X3BlockDiagonal,
    inverse_2x2_block_diagonal,
    inv2X2BlockDiagonal,
    TensorType,
    make_property_tensor,
    makePropertyTensor,
    inverse_property_tensor,
    invPropertyTensor,
    estimate_diagonal,
    diagEst,
    Zero,
    Identity,
    uniqueRows,
    eigenvalue_by_power_iteration,
    cartesian2spherical,
    spherical2cartesian,
    coterminal,
    define_plane_from_points,
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
from .mesh_utils import (
    meshTensor,
    closestPoints,
    ExtractCoreMesh,
    unpack_widths,
    closest_points_index,
    extract_core_mesh
)
from .curv_utils import (
    volTetra,
    faceInfo,
    indexCube,
    ExampleLrmGrid,
    volume_tetrahedron,
    index_cube,
    face_info,
    example_curvilinear_grid
)
from .counter_utils import Counter, count, timeIt
from . import model_builder
from . import solver_utils
from . import io_utils
from .coord_utils import (
    rotatePointsFromNormals,
    rotationMatrixFromNormals,
    rotation_matrix_from_normals,
    rotate_points_from_normals
)
from .model_utils import surface2ind_topo, depth_weighting
from .plot_utils import plot2Ddata, plotLayer, plot_1d_layer_model
from .io_utils import download
from .pgi_utils import (
    make_SimplePGI_regularization,
    make_PGI_regularization,
    make_SimplePGIwithRelationships_regularization,
    GaussianMixture,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
)

"""
Deprecated,
don't think we can throw warning if a user accesses them from here...
"""
SolverUtils = solver_utils
ModelBuilder = model_builder
