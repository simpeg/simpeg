from discretize.tests import assert_expected_order, check_derivative
from discretize.utils import example_simplex_mesh
import discretize
import numpy as np
from SimPEG.regularization import SmoothnessFullGradient
import pytest


def f_2d(x, y):
    return (1 - np.cos(2 * x * np.pi)) * (1 - np.cos(4 * y * np.pi))


def f_3d(x, y, z):
    return f_2d(x, y) * (1 - np.cos(8 * z * np.pi))


dir_2d = np.array([[1.0, 1.0], [-1.0, 1.0]]).T
dir_2d /= np.linalg.norm(dir_2d, axis=0)
dir_3d = np.array([[1, 1, 1], [-1, 1, 0], [-1, -1, 2]]).T
dir_3d = dir_3d / np.linalg.norm(dir_3d, axis=0)

# a list of argument tuples to pass to pytest parameterize
# each is a tuple of (function, dim, true_value, alphas, reg_dirs)
parameterized_args = [
    (f_2d, 2, 15 * np.pi**2, [1, 1], None),  # assumes reg_dirs aligned with axes
    (
        f_2d,
        2,
        15 * np.pi**2,
        [1, 1],
        np.eye(2),
    ),  # test for explicitly aligned with axes
    (
        f_2d,
        2,
        15 * np.pi**2,
        [1, 1],
        dir_2d,
    ),  # circular regularization should be invariant to rotation
    (
        f_2d,
        2,
        27 * np.pi**2,
        [1, 2],
        None,
    ),  # elliptic regularization aligned with axes
    (f_2d, 2, 111.033049512255 * 2, [1, 2], dir_2d),  # rotated elliptic regularization
    (
        f_3d,
        3,
        189 * np.pi**2 / 2,
        [1, 1, 1],
        None,
    ),  # test for explicitly aligned with axes
    (
        f_3d,
        3,
        189 * np.pi**2 / 2,
        [1, 1, 1],
        np.eye(3),
    ),  # test for explicitly aligned with axes
    (
        f_3d,
        3,
        189 * np.pi**2 / 2,
        [1, 1, 1],
        dir_3d,
    ),  # circular regularization should be invariant to rotation
    (
        f_3d,
        3,
        513 * np.pi**2 / 2,
        [1, 2, 3],
        None,
    ),  # elliptic regularization aligned with axes
    (
        f_3d,
        3,
        1065.91727531765 * 2,
        [1, 2, 3],
        dir_3d,
    ),  # rotated elliptic regularization
]


@pytest.mark.parametrize("mesh_class", [discretize.TensorMesh, discretize.TreeMesh])
@pytest.mark.parametrize("func,dim,true_value,alphas,reg_dirs", parameterized_args)
def test_regulariation_order(mesh_class, func, dim, true_value, alphas, reg_dirs):
    """This function is testing for the accuracy of the regularization.
    Basically, is it actually measuring what we say it's measuring.
    """
    n_hs = [8, 16, 32]

    def reg_error(n):
        h = [n] * dim
        mesh = mesh_class(h)
        if mesh_class is discretize.TreeMesh:
            mesh.refine(-1)
        # cell widths will be the same in each dimension
        dh = mesh.h[0][0]

        f_eval = func(*mesh.cell_centers.T)

        reg = SmoothnessFullGradient(mesh, alphas=alphas, reg_dirs=reg_dirs)

        numerical_eval = reg(f_eval)
        err = np.abs(numerical_eval - true_value)
        return err, dh

    assert_expected_order(reg_error, n_hs)


@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_mesh(dim):
    """Test to make sure it works with a simplex mesh

    We can't make as strong of an accuracy claim for this mesh type because the cell gradient
    operator is not actually defined for it (it uses an approximation to the cell gradient).
    It is close, but we should at least test that it works..
    """
    h = [10] * dim
    points, simplices = example_simplex_mesh(h)
    mesh = discretize.SimplexMesh(points, simplices)
    reg = SmoothnessFullGradient(mesh)

    # multiply it by a vector to make sure we can construct everything internally
    # at the very least, we should be able to confirm it evaluates to 0 for a flat model.
    out = reg(np.ones(mesh.n_cells))
    np.testing.assert_allclose(out, 0)


@pytest.mark.parametrize(
    "dim,alphas,reg_dirs", [(2, [1, 2], dir_2d), (3, [1, 2, 3], dir_3d)]
)
def test_first_derivatives(dim, alphas, reg_dirs):
    """Perform a derivative test."""
    h = [10] * dim
    mesh = discretize.TensorMesh(h)
    reg = SmoothnessFullGradient(mesh, alphas=alphas, reg_dirs=reg_dirs)

    def func(x):
        return reg(x), reg.deriv(x)

    check_derivative(func, np.ones(mesh.n_cells), plotIt=False)


@pytest.mark.parametrize(
    "dim,alphas,reg_dirs", [(2, [1, 2], dir_2d), (3, [1, 2, 3], dir_3d)]
)
def test_second_derivatives(dim, alphas, reg_dirs):
    """Perform a derivative test."""
    h = [10] * dim
    mesh = discretize.TensorMesh(h)
    reg = SmoothnessFullGradient(mesh, alphas=alphas, reg_dirs=reg_dirs)

    def func(x):
        return reg.deriv(x), lambda v: reg.deriv2(x, v)

    check_derivative(func, np.ones(mesh.n_cells), plotIt=False)


@pytest.mark.parametrize("with_active_cells", [True, False])
def test_operations(with_active_cells, dim=3):
    # Here we just make sure operations at least work
    h = [10] * dim
    mesh = discretize.TensorMesh(h)
    if with_active_cells:
        active_cells = mesh.cell_centers[:, -1] <= 0.75
        n_cells = active_cells.sum()
    else:
        active_cells = None
        n_cells = mesh.n_cells
    reg = SmoothnessFullGradient(mesh, active_cells=active_cells)
    # create a model
    m = np.arange(n_cells)
    # create a vector
    v = np.random.rand(n_cells)
    # test the second derivative evaluates
    # and gives same results with and without a vector
    v1 = reg.deriv2(m, v)
    v2 = reg.deriv2(m) @ v
    np.testing.assert_allclose(v1, v2)

    W1 = reg.W

    # test assigning n_cells
    reg.set_weights(temp_weight=np.random.rand(n_cells))

    # setting a weight should've erased W
    assert reg._W is None

    # test assigning n_total_faces face weight
    reg.set_weights(temp_weight=np.random.rand(mesh.n_faces))

    # and test it all works!
    W2 = reg.W
    assert W1 is not W2


def test_errors():
    # bad dimension mesh
    mesh1d = discretize.TensorMesh([5])
    with pytest.raises(TypeError):
        SmoothnessFullGradient(mesh1d)
    mesh2d = discretize.TensorMesh([5, 5])
    # test some bad alphas
    with pytest.raises(ValueError):
        # 3D alpha passed to 2D operator
        SmoothnessFullGradient(mesh2d, [1, 2, 3])

    with pytest.raises(IndexError):
        # incorrect number cell dependent alphas
        alphas = np.random.rand(mesh2d.n_cells - 5, 2)
        SmoothnessFullGradient(mesh2d, alphas=alphas)

    with pytest.raises(ValueError):
        # negative alphas
        SmoothnessFullGradient(mesh2d, [-1, 1, 1])

    alphas = [1, 2]
    # test some bad reg dirs
    with pytest.raises(ValueError):
        # 3D reg dirs to 2D reg
        reg_dirs = np.random.rand(3, 3)
        SmoothnessFullGradient(mesh2d, alphas=alphas, reg_dirs=reg_dirs)

    with pytest.raises(IndexError):
        # incorrect number of cell dependent reg_dirs
        reg_dirs = np.random.rand(mesh2d.n_cells - 5, 2, 2)
        SmoothnessFullGradient(mesh2d, alphas=alphas, reg_dirs=reg_dirs)

    with pytest.raises(ValueError):
        # non orthnormal reg_dirs
        # incorrect number of cell dependent reg_dirs
        reg_dirs = np.random.rand(2, 2)
        SmoothnessFullGradient(mesh2d, alphas=alphas, reg_dirs=reg_dirs)
