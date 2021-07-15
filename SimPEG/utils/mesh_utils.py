from discretize.utils import unpack_widths, closest_points_index, extract_core_mesh
from .code_utils import deprecate_method, deprecate_function


################################################
#             DEPRECATED FUNCTIONS
################################################

meshTensor = deprecate_method(
    unpack_widths, "meshTensor", removal_version="0.16.0", future_warn=True
)

closestPoints = deprecate_method(
    closest_points_index, "closestPoints", removal_version="0.16.0", future_warn=True
)

ExtractCoreMesh = deprecate_method(
    extract_core_mesh, "ExtractCoreMesh", removal_version="0.16.0", future_warn=True
)