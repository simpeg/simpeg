"""
========================================================
Utility Classes and Functions (:mod:`simpeg.utils`)
========================================================
.. currentmodule:: simpeg.utils

The ``utils`` package contains utilities for helping with common operations involving
SimPEG.

Many of the utilities are imported from `discretize.utils`. See that package's
documentation for many details on items.


Counter Utility Functions
=========================

.. autosummary::
  :toctree: generated/

  Counter
  count
  timeIt


IO Utility Functions
====================

.. autosummary::
  :toctree: generated/

  download
  io_utils.read_dcip2d_ubc
  io_utils.read_dcip3d_ubc
  io_utils.read_dcipoctree_ubc
  io_utils.read_dcip_xyz
  io_utils.read_gg3d_ubc
  io_utils.read_grav3d_ubc
  io_utils.read_mag3d_ubc
  io_utils.write_dcip2d_ubc
  io_utils.write_dcip3d_ubc
  io_utils.write_dcipoctree_ubc
  io_utils.write_dcip_xyz
  io_utils.write_grav3d_ubc
  io_utils.write_gg3d_ubc
  io_utils.write_mag3d_ubc


Matrix Utility Functions
========================

.. autosummary::
  :toctree: generated/

  cartesian2spherical
  coterminal
  define_plane_from_points
  eigenvalue_by_power_iteration
  estimate_diagonal
  spherical2cartesian
  unique_rows


Mesh Utility Functions
======================

.. autosummary::
  :toctree: generated/

  surface2inds


Model Utility Functions
=======================

.. autosummary::
  :toctree: generated/

  depth_weighting
  model_builder.add_block
  model_builder.create_2_layer_model
  model_builder.create_block_in_wholespace
  model_builder.create_ellipse_in_wholespace
  model_builder.create_from_function
  model_builder.create_layers_model
  model_builder.create_random_model
  model_builder.get_indices_block
  model_builder.get_indices_polygon
  model_builder.get_indices_sphere


Plotting Utility Functions
==========================

.. autosummary::
  :toctree: generated/

  plot2Ddata
  plot_1d_layer_model


PGI Utility Classes and Functions
=================================
.. autosummary::
  :toctree: generated/

  WeightedGaussianMixture
  GaussianMixtureWithPrior
  GaussianMixtureWithNonlinearRelationships
  GaussianMixtureWithNonlinearRelationshipsWithPrior

Code Utility Functions
======================
Many of the functions here are used internally to SimPEG and have minimal documentation.

.. autosummary::
  :toctree: generated/

  call_hooks
  check_stoppers
  mem_profile_class
  dependent_property
  deprecate_class
  deprecate_function
  deprecate_method
  deprecate_module
  deprecate_property
  hook
  print_done
  print_line
  print_stoppers
  print_titles
  requires
  set_kwargs
  validate_float
  validate_integer
  validate_list_of_types
  validate_location_property
  validate_ndarray_with_shape
  validate_string
  validate_callable
  validate_direction
  validate_active_indices

"""

from discretize.utils.interpolation_utils import interpolation_matrix

from .code_utils import (
    mem_profile_class,
    hook,
    set_kwargs,
    print_titles,
    print_line,
    check_stoppers,
    print_stoppers,
    print_done,
    call_hooks,
    deprecate_property,
    deprecate_module,
    deprecate_method,
    deprecate_function,
    deprecate_class,
    dependent_property,
    as_array_n_by_dim,
    requires,
    Report,
    validate_float,
    validate_integer,
    validate_list_of_types,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_string,
    validate_type,
    validate_callable,
    validate_direction,
    validate_active_indices,
)

from .mat_utils import (
    mkvc,
    sdiag,
    sdinv,
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
    inverse_3x3_block_diagonal,
    inverse_2x2_block_diagonal,
    TensorType,
    make_property_tensor,
    inverse_property_tensor,
    estimate_diagonal,
    Zero,
    Identity,
    unique_rows,
    eigenvalue_by_power_iteration,
    cartesian2spherical,
    spherical2cartesian,
    coterminal,
    define_plane_from_points,
)
from .mesh_utils import (
    unpack_widths,
    closest_points_index,
    extract_core_mesh,
    surface2inds,
)
from .curv_utils import (
    volume_tetrahedron,
    index_cube,
    face_info,
    example_curvilinear_grid,
)
from .counter_utils import Counter, count, timeIt
from . import model_builder
from . import solver_utils
from . import io_utils
from .coord_utils import (
    rotation_matrix_from_normals,
    rotate_points_from_normals,
)
from .model_utils import depth_weighting
from .plot_utils import plot2Ddata, plotLayer, plot_1d_layer_model
from .io_utils import download
from .pgi_utils import (
    GaussianMixture,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
)

# Deprecated imports
interpmat = deprecate_function(
    interpolation_matrix, "interpmat", removal_version="0.19.0", error=True
)

from .code_utils import (
    memProfileWrapper,
    setKwargs,
    printTitles,
    printLine,
    checkStoppers,
    printStoppers,
    printDone,
    callHooks,
    dependentProperty,
    asArray_N_x_Dim,
)
from .mat_utils import (
    sdInv,
    getSubArray,
    inv3X3BlockDiagonal,
    inv2X2BlockDiagonal,
    makePropertyTensor,
    invPropertyTensor,
    diagEst,
    uniqueRows,
)
from .mesh_utils import (
    meshTensor,
    closestPoints,
    ExtractCoreMesh,
)
from .curv_utils import (
    volTetra,
    faceInfo,
    indexCube,
    exampleLrmGrid,
)
from .coord_utils import (
    rotatePointsFromNormals,
    rotationMatrixFromNormals,
)
