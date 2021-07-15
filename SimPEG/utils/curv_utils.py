from discretize.utils import volume_tetrahedron, index_cube, face_info, example_curvilinear_grid

################################################
#             DEPRECATED FUNCTIONS
################################################

volTetra = deprecate_method(
    volume_tetrahedron, "volTetra", removal_version="0.16.0", future_warn=True
)

indexCube = deprecate_method(
    index_cube, "indexCube", removal_version="0.16.0", future_warn=True
)

faceInfo = deprecate_method(
    face_info, "faceInfo", removal_version="0.16.0", future_warn=True
)

ExampleLrmGrid = deprecate_method(
    example_curvilinear_grid, "ExampleLrmGrid", removal_version="0.16.0", future_warn=True
)
