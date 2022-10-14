"""
========================================================
Utility Classes and Functions (:mod:`SimPEG.utils`)
========================================================
.. currentmodule:: SimPEG.utils

The ``utils`` package contains utilities for helping with common operations involving
SimPEG.

Many of the utilities are imported from `discretize.utils`. See that package's
documentation for many details on items.


Coordinates Utility Functions
=============================

.. autosummary::
  :toctree: generated/

  rotation_matrix_from_normals
  rotate_points_from_normals

Counter Utility Functions
=========================

.. autosummary::
  :toctree: generated/

  Counter
  count
  timeIt

Curvilinear Utility Functions
=============================

.. autosummary::
  :toctree: generated/

  example_curvilinear_grid
  face_info
  index_cube
  volume_tetrahedron


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

  av
  av_extrap
  cartesian2spherical
  coterminal
  ddx
  define_plane_from_points
  diagEst
  eigenvalue_by_power_iteration
  estimate_diagonal
  get_subarray
  kron3
  ind2sub
  inverse_2x2_block_diagonal
  inverse_3x3_block_diagonal
  inverse_property_tensor
  make_property_tensor
  mkvc
  ndgrid
  sdiag
  sdinv
  speye
  spherical2cartesian
  spzeros
  sub2ind
  unique_rows


Mesh Utility Functions
======================

.. autosummary::
  :toctree: generated/

  closest_points_index
  extract_core_mesh
  unpack_widths
  surface2inds


Model Utility Functions
=======================

.. autosummary::
  :toctree: generated/

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

  asArray_N_x_Dim
  call_hooks
  check_stoppers
  create_wrapper_from_class
  dependent_property
  deprecate_class
  deprecate_function
  deprecate_method
  deprecate_module
  deprecate_property
  hook
  print_done
  printDone
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

"""
from __future__ import print_function

from discretize.utils.interpolation_utils import interpmat

from .code_utils import (
    create_wrapper_from_class,
    memProfileWrapper,
    hook,
    set_kwargs,
    setKwargs,
    print_titles,
    printTitles,
    print_line,
    printLine,
    check_stoppers,
    checkStoppers,
    print_stoppers,
    printStoppers,
    print_done,
    printDone,
    call_hooks,
    callHooks,
    deprecate_property,
    deprecate_module,
    deprecate_method,
    deprecate_function,
    deprecate_class,
    dependent_property,
    dependentProperty,
    asArray_N_x_Dim,
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
)

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
    unique_rows,
    uniqueRows,
    eigenvalue_by_power_iteration,
    cartesian2spherical,
    spherical2cartesian,
    coterminal,
    define_plane_from_points,
)
from .mesh_utils import (
    meshTensor,
    closestPoints,
    ExtractCoreMesh,
    unpack_widths,
    closest_points_index,
    extract_core_mesh,
    surface2inds,
)
from .curv_utils import (
    volTetra,
    faceInfo,
    indexCube,
    exampleLrmGrid,
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
    rotatePointsFromNormals,
    rotationMatrixFromNormals,
    rotation_matrix_from_normals,
    rotate_points_from_normals,
)
from .model_utils import surface2ind_topo, depth_weighting
from .plot_utils import plot2Ddata, plotLayer, plot_1d_layer_model
from .io_utils import download
from .pgi_utils import (
    GaussianMixture,
    WeightedGaussianMixture,
    GaussianMixtureWithPrior,
    GaussianMixtureWithNonlinearRelationships,
    GaussianMixtureWithNonlinearRelationshipsWithPrior,
)
