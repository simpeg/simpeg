"""
========================================================
Utility Classes and Functions (:mod:`SimPEG.utils`)
========================================================
.. currentmodule:: SimPEG.utils

The ``utils`` package contains utilities for helping with common operations involving
SimPEG.

Utility Classes
===============
.. autosummary::
  :toctree: generated/

  Counter
  Identity
  Report
  TensorType
  Zero
  

Utility Functions
=================

Code Utilities
--------------
.. autosummary::
  :toctree: generated/

  asArray_N_x_Dim
  call_hooks
  callHooks
  check_stoppers
  checkStoppers
  create_wrapper_from_class
  dependent_property
  dependentProperty
  hook
  memProfileWrapper
  print_done
  printDone
  print_line
  printLine
  print_stoppers
  printStoppers
  print_titles
  printTitles
  requires
  set_kwargs
  setKwargs

Coordinates Utilities
---------------------

.. autosummary::
  :toctree: generated/

  rotatePointsFromNormals
  rotationMatrixFromNormals
  rotation_matrix_from_normals
  rotate_points_from_normals

Counter Utilities
-----------------

.. autosummary::
  :toctree: generated/

  count
  timeIt

Curvilinear Utilities
---------------------

.. autosummary::
  :toctree: generated/

  example_curvilinear_grid
  ExampleLrmGrid
  face_info
  faceInfo
  index_cube
  indexCube
  volume_tetrahedron
  volTetra
  

IO Utilities
------------

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
  

Matrix Utilities
----------------

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
  getSubArray
  kron3
  ind2sub
  inverse_2x2_block_diagonal
  inv2X2BlockDiagonal
  inverse_3x3_block_diagonal
  inv3X3BlockDiagonal
  inverse_property_tensor
  invPropertyTensor
  make_property_tensor
  makePropertyTensor
  mkvc
  ndgrid
  sdiag
  sdinv
  speye
  spherical2cartesian
  spzeros
  sub2ind
  uniqueRows
  

Mesh Utilities
--------------

.. autosummary::
  :toctree: generated/

  closest_points_index
  closestPoints
  extract_core_mesh
  ExtractCoreMesh
  meshTensor
  unpack_widths
  

Model Builder Utilities
-----------------------

.. autosummary::
  :toctree: generated/

  model_builder.add_block
  model_builder.addBlock
  model_builder.create_2_layer_model
  model_builder.create_block_in_wholespace
  model_builder.create_ellipse_in_wholespace
  model_builder.create_from_function
  model_builder.create_layers_model
  model_builder.create_random_model
  model_builder.defineBlock
  model_builder.defineEllipse
  model_builder.defineTwoLayers
  model_builder.get_indices_block
  model_builder.getIndicesBlock
  model_builder.get_indices_polygon
  model_builder.get_indices_sphere
  model_builder.getIndicesSphere
  model_builder.layeredModel
  model_builder.polygonInd
  model_builder.randomModel
  model_builder.scalarConductivity
  

Plotting Utilities
------------------

.. autosummary::
  :toctree: generated/

  plot2Ddata
  plotLayer
  plot_1d_layer_model

"""
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
    dependent_property,
    dependentProperty,
    asArray_N_x_Dim,
    requires,
    Report
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
