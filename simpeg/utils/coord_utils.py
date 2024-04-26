from discretize.utils import (  # noqa: F401
    rotation_matrix_from_normals,
    rotate_points_from_normals,
)
from .code_utils import deprecate_function

# deprecated functions
rotationMatrixFromNormals = deprecate_function(
    rotation_matrix_from_normals,
    "rotationMatrixFromNormals",
    removal_version="0.19.0",
    error=True,
)
rotatePointsFromNormals = deprecate_function(
    rotate_points_from_normals,
    "rotatePointsFromNormals",
    removal_version="0.19.0",
    error=True,
)
