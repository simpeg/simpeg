from discretize.utils import rotation_matrix_from_normals, rotate_points_from_normals

################################################
#             DEPRECATED FUNCTIONS
################################################

rotationMatrixFromNormals = deprecate_method(
    rotation_matrix_from_normals, "rotationMatrixFromNormals", removal_version="0.16.0", future_warn=True
)

rotatePointsFromNormals = deprecate_method(
    rotate_points_from_normals, "rotatePointsFromNormals", removal_version="0.16.0", future_warn=True
)