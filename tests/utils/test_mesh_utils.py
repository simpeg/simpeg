import pytest
import numpy as np
from discretize import TensorMesh, TreeMesh, SimplexMesh
from discretize.utils import example_simplex_mesh
from simpeg.utils import shift_to_discrete_topography, get_discrete_topography

DH = 1.0
N = 16


def get_mesh(mesh_type, dim):
    """Generate test mesh."""
    h = dim * [[(DH, N)]]
    origin = dim * "C"
    if mesh_type == "tensor":
        return TensorMesh(h, origin)
    elif mesh_type == "tree":
        tree_mesh = TreeMesh(h, origin)
        tree_mesh.refine(-1)
        return tree_mesh
    else:
        points, simplices = example_simplex_mesh(dim * [N])
        points = N * points - N / 2
        return SimplexMesh(points, simplices)


def get_points(dim):
    """Test points."""
    if dim == 2:
        return np.array([[1.1, 3.0], [-3.9, -2.0]])
    else:
        return np.array([[1.1, -3.6, 3.0], [-3.9, 4.4, -2.0]])


def get_active_cells(mesh):
    """Test active cells for the mesh."""
    active_cells = np.zeros(mesh.n_cells, dtype=bool)
    if mesh.dim == 1:
        active_cells[mesh.cell_centers < 0.0] = True
    else:
        active_cells[mesh.cell_centers[:, -1] < 0.0] = True
    return active_cells


CASES_LIST_SUCCESS = [
    ("tensor", 2, "center", False, 0.0),
    ("tensor", 3, "top", False, np.r_[1.25, 1.25]),
    ("tree", 2, "top", False, np.r_[1.25, 1.25]),
    ("tree", 3, "center", False, 0.0),
    ("tensor", 2, "center", True, 0.0),
    ("tensor", 3, "top", True, np.r_[1.25, 1.25]),
    ("tree", 2, "top", True, np.r_[1.25, 1.25]),
    ("tree", 3, "center", True, 0.0),
]


@pytest.mark.parametrize(
    "mesh_type, dim, option, shift_horizontal, heights", CASES_LIST_SUCCESS
)
def test_function_success(mesh_type, dim, option, shift_horizontal, heights):
    """Test cases that run properly."""
    mesh = get_mesh(mesh_type, dim)
    active_cells = get_active_cells(mesh)
    pts = get_points(dim)

    pts_shifted = shift_to_discrete_topography(
        mesh,
        pts,
        active_cells,
        topo_cell_cutoff=option,
        shift_horizontal=shift_horizontal,
        heights=heights,
    )

    if isinstance(heights, (int, float)):
        heights = np.array([heights])

    correct_elevations = heights.copy()
    if option == "center":
        correct_elevations -= 0.5 * DH

    if shift_horizontal:
        correct_locations = np.round(pts) + 0.5 * DH
    else:
        correct_locations = pts.copy()
    correct_locations[:, -1] = correct_elevations

    np.testing.assert_allclose(correct_locations, pts_shifted)


def test_mesh_type_error():
    """Throw unsupported mesh type error."""
    mesh_type = "simplex"
    dim = 2

    mesh = get_mesh(mesh_type, dim)
    active_cells = get_active_cells(mesh)
    pts = get_points(dim)

    with pytest.raises(NotImplementedError):
        shift_to_discrete_topography(mesh, pts, active_cells)

    with pytest.raises(NotImplementedError):
        get_discrete_topography(mesh, active_cells)


def test_size_errors():
    """Throw array size mismatch error."""
    mesh_type = "tensor"
    dim = 3

    mesh = get_mesh(mesh_type, dim)
    active_cells = get_active_cells(mesh)
    pts = get_points(dim)
    heights = np.r_[1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        shift_to_discrete_topography(mesh, pts, active_cells, heights=heights)


# mesh_type = "tensor"
# dim = 3
# option = "top"
# shift_horizontal = True
# heights = 0.0
# test_function_success(mesh_type, dim, option, shift_horizontal, heights)
