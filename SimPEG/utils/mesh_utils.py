from discretize.utils import unpack_widths, closest_points_index, extract_core_mesh
from .code_utils import deprecate_method, deprecate_function


################################################
#             DEPRECATED FUNCTIONS
################################################

meshTensor = deprecate_function(
    unpack_widths, "meshTensor", removal_version="0.16.0"
)

closestPoints = deprecate_function(
    closest_points_index, "closestPoints", removal_version="0.16.0"
)

ExtractCoreMesh = deprecate_function(
    extract_core_mesh, "ExtractCoreMesh", removal_version="0.16.0"
)