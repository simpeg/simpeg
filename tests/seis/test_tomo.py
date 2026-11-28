import re
import unittest

import discretize
import matplotlib.pyplot as plt
import numpy as np
import pytest

from simpeg import maps, tests, utils
from simpeg.seismic import straight_ray_tomography as tomo
from simpeg.seismic.straight_ray_tomography.simulation import (
    Simulation2DIntegral,
    SimulationIntegral,
    _lengthInCell,
    _lineintegral,
)

TOL = 1e-5
FLR = 1e-14


class TomoTest(unittest.TestCase):
    def setUp(self):
        nC = 20
        M = discretize.TensorMesh([nC, nC])
        y = np.linspace(0.0, 1.0, nC // 2)
        rlocs = np.c_[y * 0 + M.cell_centers_x[-1], y]
        rx = tomo.Rx(locations=rlocs)
        source_list = [
            tomo.Src(location=np.r_[M.cell_centers_x[0], yi], receiver_list=[rx])
            for yi in y
        ]
        survey = tomo.Survey(source_list)
        problem = tomo.Simulation(M, survey=survey, slownessMap=maps.IdentityMap(M))
        self.M = M
        self.problem = problem
        self.survey = survey

    def test_deriv(self):
        s = utils.mkvc(utils.model_builder.create_random_model(self.M.vnC)) + 1.0

        def fun(x):
            return self.problem.dpred(x), lambda x: self.problem.Jvec(s, x)

        return tests.check_derivative(
            fun, s, num=4, plotIt=False, eps=FLR, random_seed=664
        )


@pytest.mark.parametrize(
    "origin, receiver, bounds, expected",
    [
        ([0.25, 0.5], [1.75, 0.5], [[0.0, 1.0], [0.0, 1.0]], 0.75),
        (
            [0.25, 0.5, 0.5],
            [1.75, 0.5, 0.5],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            0.75,
        ),
    ],
)
def test_length_in_cell_2d_and_3d(origin, receiver, bounds, expected):
    direction = np.asarray(receiver) - np.asarray(origin)
    assert _lengthInCell(origin, direction, bounds) == pytest.approx(expected)


def test_length_in_cell_rejects_zero_length_ray():
    assert _lengthInCell([0.5, 0.5], [0.0, 0.0], [[0.0, 1.0], [0.0, 1.0]]) is None


@pytest.mark.parametrize(
    "origin, direction, bounds, message",
    [
        ([0.0, 0.0], [1.0], [[0.0, 1.0], [0.0, 1.0]], "one-dimensional"),
        ([0.0, 0.0], [1.0, 0.0], [[0.0, 1.0]], "shape"),
    ],
)
def test_length_in_cell_rejects_invalid_input_shapes(
    origin, direction, bounds, message
):
    with pytest.raises(ValueError, match=message):
        _lengthInCell(origin, direction, bounds)


def test_length_in_cell_plotting_2d_and_3d():
    plt.close("all")
    assert _lengthInCell(
        [0.25, 0.5], [1.5, 0.0], [[0.0, 1.0], [0.0, 1.0]], plotIt=True
    ) == pytest.approx(0.75)
    plt.close("all")

    # With no current 3D axes, the helper creates one for the debug plot.
    assert _lengthInCell(
        [0.25, 0.5, 0.5],
        [1.5, 0.0, 0.0],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        plotIt=True,
    ) == pytest.approx(0.75)
    plt.close("all")

    # It should also draw into an already active 3D axes.
    fig = plt.figure()
    fig.add_subplot(111, projection="3d")
    assert _lengthInCell(
        [0.25, 0.5, 0.5],
        [1.5, 0.0, 0.0],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        plotIt=True,
    ) == pytest.approx(0.75)
    plt.close(fig)


def test_lineintegral_rejects_unsupported_mesh_dimension():
    mesh = discretize.TensorMesh([[1.0, 1.0]])
    with pytest.raises(ValueError, match="requires a 2D or 3D mesh, got 1D"):
        _lineintegral(mesh, [0.0], [1.0])


def test_lineintegral_returns_empty_for_ray_outside_mesh():
    mesh = discretize.TensorMesh([[1.0], [1.0]])
    inds, lengths = _lineintegral(mesh, [0.0, 2.0], [1.0, 2.0])

    assert inds.size == 0
    assert lengths.size == 0


def test_lineintegral_and_simulation_support_3d_mesh():
    mesh = discretize.TensorMesh([[1.0, 1.0], [1.0], [1.0]])
    inds, lengths = _lineintegral(mesh, [0.0, 0.5, 0.5], [2.0, 0.5, 0.5])

    np.testing.assert_array_equal(inds, [0, 1])
    np.testing.assert_allclose(lengths, [1.0, 1.0])
    receiver = tomo.Rx(locations=np.array([[2.0, 0.5, 0.5]]))
    source = tomo.Src(location=np.array([0.0, 0.5, 0.5]), receiver_list=[receiver])
    simulation = SimulationIntegral(
        mesh,
        survey=tomo.Survey([source]),
        slownessMap=maps.IdentityMap(mesh),
    )
    assert simulation.mesh is mesh
    np.testing.assert_allclose(simulation.A.toarray(), [[1.0, 1.0]])
    np.testing.assert_allclose(
        simulation.Jtvec(np.ones(mesh.nC), np.ones(1)), [1.0, 1.0]
    )


def test_required_mesh_arg():
    msg = ".*missing 1 required positional argument: 'mesh'"
    with pytest.raises(TypeError, match=msg):
        Simulation2DIntegral()


def test_bad_mesh_type():
    mesh = discretize.CylindricalMesh([3, 3, 3])
    msg = "mesh must be an instance of TensorMesh, not CylindricalMesh"
    with pytest.raises(TypeError, match=msg):
        Simulation2DIntegral(mesh)


def test_bad_mesh_dim():
    mesh = discretize.TensorMesh([3, 3, 3])
    msg = re.escape("Simulation2DIntegral mesh must be 2D, received a 3D mesh.")
    with pytest.raises(ValueError, match=msg):
        Simulation2DIntegral(mesh)


if __name__ == "__main__":
    unittest.main()
