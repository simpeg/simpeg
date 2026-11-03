import re
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

    msg = re.escape(
        f"Invalid mesh of type '{type(mesh).__name__}'. "
        "`shift_to_discrete_topography` only supports TensorMesh and TreeMesh."
    )
    with pytest.raises(TypeError, match=msg):
        shift_to_discrete_topography(mesh, pts, active_cells)

    msg = re.escape(
        f"Invalid mesh of type '{type(mesh).__name__}'. "
        "`get_discrete_topography` only supports TensorMesh and TreeMesh."
    )
    with pytest.raises(TypeError, match=msg):
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


@pytest.mark.parametrize("shift_horizontal", [False, True])
def test_tree_mesh_own_column(shift_horizontal):
    """On a tree mesh, the shifted locations are on the discrete surface.

    The surface steps from z = 0 (x < 0, 1 m cells) down to z = -2 (x > 0, 2 m
    cells), so the surface cell spanning 0 < x < 2 has its center at x = 1, further
    from x = 0.2 than the 1 m cell centered at x = -0.5. Without the horizontal shift
    x = 0.2 keeps the surface of its own column, with it x = 0.2 moves to the
    closest surface cell center (and that cell's top).
    """
    mesh = TreeMesh([np.ones(32), np.ones(32)], "CC", diagonal_balance=True)
    mesh.refine(
        lambda cell: mesh.max_level - (cell.center[0] > 0 or abs(cell.center[1]) > 2),
        diagonal_balance=True,
    )
    x, z = mesh.cell_centers.T
    active_cells = z < np.where(x < 0, 0.0, -2.0)
    pts = np.array([[0.2, 5.0], [-0.2, 5.0]])

    pts_shifted = shift_to_discrete_topography(
        mesh, pts, active_cells, shift_horizontal=shift_horizontal
    )
    if shift_horizontal:
        expected = [[-0.5, 0.0], [-0.5, 0.0]]
    else:
        expected = [[0.2, -2.0], [-0.2, 0.0]]
    np.testing.assert_allclose(pts_shifted, expected)


def test_integer_active_cells():
    mesh = get_mesh("tensor", 3)
    active_cells = get_active_cells(mesh)
    pts = get_points(3)
    np.testing.assert_allclose(
        shift_to_discrete_topography(mesh, pts, np.flatnonzero(active_cells)),
        shift_to_discrete_topography(mesh, pts, active_cells),
    )


def test_invalid_topo_cell_cutoff():
    mesh = get_mesh("tree", 2)
    with pytest.raises(ValueError, match="'top' or 'center'"):
        shift_to_discrete_topography(
            mesh, get_points(2), get_active_cells(mesh), topo_cell_cutoff="bottom"
        )


def test_no_active_cells_below():
    mesh = get_mesh("tensor", 2)
    active_cells = get_active_cells(mesh)
    active_cells[mesh.cell_centers[:, 0] > 0] = False
    with pytest.raises(ValueError, match="No active cells below"):
        shift_to_discrete_topography(
            mesh, get_points(2), active_cells, shift_horizontal=False
        )


def test_discrete_topography_split_under_finer_cells():
    """A coarse surface cell under finer inactive cells is split up as those cells."""
    # 1 m cells in the air above 0 < x < 2, 2 m cells everywhere else
    mesh = TreeMesh([np.ones(8), np.ones(8)], "CC", diagonal_balance=True)

    def level(cell):
        x, z = cell.center
        return mesh.max_level if (0 < x < 2 and z > 0) else mesh.max_level - 1

    mesh.refine(level, diagonal_balance=True)
    active_cells = mesh.cell_centers[:, 1] < 0
    topo = get_discrete_topography(mesh, active_cells)
    np.testing.assert_allclose(
        topo[(topo[:, 0] > 0) & (topo[:, 0] < 2)], [[0.5, 0], [1.5, 0]]
    )


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_discrete_topography_upper_surface(mesh_type):
    """Only the upper surface, not the floor of an inactive pocket below it."""
    mesh = get_mesh(mesh_type, 2)
    x, z = mesh.cell_centers.T
    active_cells = (z < 0) & ~((np.abs(x - 0.5) < 0.6) & (np.abs(z + 2.5) < 0.6))
    topo = get_discrete_topography(mesh, active_cells)
    np.testing.assert_allclose(topo, np.c_[np.arange(-N / 2, N / 2) + 0.5, np.zeros(N)])
    pts_shifted = shift_to_discrete_topography(
        mesh, np.array([[0.7, 0.0]]), active_cells
    )
    np.testing.assert_allclose(pts_shifted, [[0.5, 0.0]])


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_discrete_topography_integer_active_cells(mesh_type):
    mesh = get_mesh(mesh_type, 3)
    active_cells = get_active_cells(mesh)
    np.testing.assert_allclose(
        get_discrete_topography(mesh, np.flatnonzero(active_cells)),
        get_discrete_topography(mesh, active_cells),
    )


def test_discrete_topography_finer_cells_higher_up():
    """Finer inactive cells above coarser inactive cells (say, refined for airborne
    data) don't give repeated surface locations."""
    # 1 m cells in a box in the air at 2 < z < 4, 2 m cells everywhere else
    mesh = TreeMesh([np.ones(16), np.ones(16)], "CC", diagonal_balance=True)

    def level(cell):
        x, z = cell.center
        return mesh.max_level if (-4 < x < 4 and 2 < z < 4) else mesh.max_level - 1

    mesh.refine(level, diagonal_balance=True)
    active_cells = mesh.cell_centers[:, 1] < 0
    topo = get_discrete_topography(mesh, active_cells)
    np.testing.assert_allclose(topo, np.c_[np.arange(-7.0, 8.0, 2.0), np.zeros(8)])


def test_discrete_topography_uneven_widths():
    """Finer inactive cells above coarser ones on a tree mesh with uneven widths."""
    h = [np.r_[2.0, 0.5, 1.5, 1.0, 0.7, 1.3, 2.5, 0.5], np.ones(8)]
    mesh = TreeMesh(h, [-h[0].sum() / 2, -4], diagonal_balance=True)

    def level(cell):
        x, z = cell.center
        return mesh.max_level if (-2 < x < 2 and 2 < z < 4) else mesh.max_level - 1

    mesh.refine(level, diagonal_balance=True)
    active_cells = mesh.cell_centers[:, 1] < 0
    topo = get_discrete_topography(mesh, active_cells)
    edges = np.cumsum(np.r_[0, h[0][0::2] + h[0][1::2]]) - h[0].sum() / 2
    np.testing.assert_allclose(topo, np.c_[0.5 * (edges[:-1] + edges[1:]), np.zeros(4)])


def test_discrete_topography_active_cell_above_air():
    """The ground under a coarse inactive cell is found even when the finer cell
    above that coarse cell's lower corner is active."""
    # 1 m cells at 4 < z < 6, 2 m cells everywhere else
    mesh = TreeMesh([np.ones(16), np.ones(16)], "CC", diagonal_balance=True)

    def level(cell):
        return mesh.max_level if 4 < cell.center[1] < 6 else mesh.max_level - 1

    mesh.refine(level, diagonal_balance=True)
    x, z = mesh.cell_centers.T
    # an active 1 m cell over the 2 m inactive cell at 0 < x < 2, 2 < z < 4
    active_cells = (z < 0) | (np.isclose(x, 0.5) & np.isclose(z, 4.5))
    topo = get_discrete_topography(mesh, active_cells)
    np.testing.assert_allclose(
        topo[(topo[:, 0] > 0) & (topo[:, 0] < 2)], [[0.5, 5.0], [1.0, 0.0]]
    )


@pytest.mark.parametrize("shift_horizontal", [False, True])
def test_shift_no_active_cells(shift_horizontal):
    mesh = get_mesh("tree", 2)
    with pytest.raises(ValueError, match="no active cells"):
        shift_to_discrete_topography(
            mesh,
            get_points(2),
            np.zeros(mesh.n_cells, dtype=bool),
            shift_horizontal=shift_horizontal,
        )


@pytest.mark.parametrize("shift_horizontal", [False, True])
def test_tensor_mesh_uneven_widths(shift_horizontal):
    """On a tensor mesh with uneven widths the closest column center isn't always
    the column a location is in.

    A 10 m column (0 < x < 10) sits next to 1 m columns, with the surface at z = 0
    for x < 10 and z = -2 beyond. The location at x = 9.6 is in the 10 m column,
    but closer to the center of the 1 m column at x = 10.5.
    """
    mesh = TensorMesh([np.r_[np.ones(10), 10.0, np.ones(10)], np.ones(10)], [-10, -5])
    x, z = mesh.cell_centers.T
    active_cells = z < np.where(x < 10, 0.0, -2.0)
    pts_shifted = shift_to_discrete_topography(
        mesh, np.array([[9.6, 5.0]]), active_cells, shift_horizontal=shift_horizontal
    )
    if shift_horizontal:
        expected = [[10.5, -2.0]]
    else:
        expected = [[9.6, 0.0]]
    np.testing.assert_allclose(pts_shifted, expected)
