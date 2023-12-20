"""
========================================================
Utility Classes and Functions (:mod:`SimPEG.utils`)
========================================================
.. currentmodule:: SimPEG.utils

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
  surface2ind_topo
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

from . import io_utils, model_builder, solver_utils
from .code_utils import (
    Report,
    as_array_n_by_dim,
    call_hooks,
    check_stoppers,
    dependent_property,
    deprecate_class,
    deprecate_function,
    deprecate_method,
    deprecate_module,
    deprecate_property,
    hook,
    mem_profile_class,
    print_done,
    print_line,
    print_stoppers,
    print_titles,
    requires,
    set_kwargs,
    validate_active_indices,
    validate_callable,
    validate_direction,
    validate_float,
    validate_integer,
    validate_list_of_types,
    validate_location_property,
    validate_ndarray_with_shape,
    validate_string,
    validate_type,
)
from .coord_utils import rotate_points_from_normals, rotation_matrix_from_normals
from .counter_utils import Counter, count, timeIt
from .curv_utils import (
    example_curvilinear_grid,
    face_info,
    index_cube,
    volume_tetrahedron,
)
from .io_utils import download
from .mat_utils import (
    Identity,
    TensorType,
    Zero,
    av,
    av_extrap,
    cartesian2spherical,
    coterminal,
    ddx,
    define_plane_from_points,
    eigenvalue_by_power_iteration,
    estimate_diagonal,
    get_subarray,
    ind2sub,
    inverse_2x2_block_diagonal,
    inverse_3x3_block_diagonal,
    inverse_property_tensor,
    kron3,
    make_property_tensor,
    mkvc,
    ndgrid,
    sdiag,
    sdinv,
    speye,
    spherical2cartesian,
    spzeros,
    sub2ind,
    unique_rows,
)
from .mesh_utils import (
    closest_points_index,
    extract_core_mesh,
    surface2inds,
    unpack_widths,
)
from .model_utils import depth_weighting, distance_weighting, surface2ind_topo
from .pgi_utils import (
    GaussianMixture,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
    GaussianMixtureWithPrior,
    WeightedGaussianMixture,
)
from .plot_utils import plot2Ddata, plot_1d_layer_model, plotLayer

# Deprecated imports
interpmat = deprecate_function(
    interpolation_matrix, "interpmat", removal_version="0.19.0", future_warn=True
)

from .code_utils import (
    asArray_N_x_Dim,
    callHooks,
    checkStoppers,
    dependentProperty,
    memProfileWrapper,
    printDone,
    printLine,
    printStoppers,
    printTitles,
    setKwargs,
)
from .coord_utils import rotatePointsFromNormals, rotationMatrixFromNormals
from .curv_utils import exampleLrmGrid, faceInfo, indexCube, volTetra
from .mat_utils import (
    diagEst,
    getSubArray,
    inv2X2BlockDiagonal,
    inv3X3BlockDiagonal,
    invPropertyTensor,
    makePropertyTensor,
    sdInv,
    uniqueRows,
)
from .mesh_utils import ExtractCoreMesh, closestPoints, meshTensor
