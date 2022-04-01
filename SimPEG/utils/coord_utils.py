from discretize.utils import rotation_matrix_from_normals, rotate_points_from_normals
from .code_utils import deprecate_function

################################################
#             DEPRECATED FUNCTIONS
################################################

rotationMatrixFromNormals = deprecate_function(
    rotation_matrix_from_normals, "rotationMatrixFromNormals", removal_version="0.16.0"
)

rotatePointsFromNormals = deprecate_function(
    rotate_points_from_normals, "rotatePointsFromNormals", removal_version="0.16.0"
)