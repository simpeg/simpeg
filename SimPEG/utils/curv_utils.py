from discretize.utils import (  # noqa: F401
    volume_tetrahedron,
    index_cube,
    face_info,
    example_curvilinear_grid,
)
from .code_utils import deprecate_function

# deprecated functions
volTetra = deprecate_function(
    volume_tetrahedron, "volTetra", removal_version="0.19.0", error=True
)
indexCube = deprecate_function(
    index_cube, "indexCube", removal_version="0.19.0", error=True
)
faceInfo = deprecate_function(
    face_info, "faceInfo", removal_version="0.19.0", error=True
)
exampleLrmGrid = deprecate_function(
    example_curvilinear_grid,
    "exampleLrmGrid",
    removal_version="0.19.0",
    error=True,
)
