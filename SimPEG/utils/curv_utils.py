from discretize.utils import volume_tetrahedron, index_cube, face_info, example_curvilinear_grid
from .code_utils import deprecate_function

################################################
#             DEPRECATED FUNCTIONS
################################################

volTetra = deprecate_function(
    volume_tetrahedron, "volTetra", removal_version="0.16.0"
)

indexCube = deprecate_function(
    index_cube, "indexCube", removal_version="0.16.0"
)

faceInfo = deprecate_function(
    face_info, "faceInfo", removal_version="0.16.0"
)

ExampleLrmGrid = deprecate_function(
    example_curvilinear_grid, "ExampleLrmGrid", removal_version="0.16.0"
)
