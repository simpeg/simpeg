import re

import numpy as np
import unittest

import discretize
import pytest

from simpeg.seismic import straight_ray_tomography as tomo
from simpeg import tests, maps, utils
from simpeg.seismic.straight_ray_tomography.simulation import Simulation2DIntegral

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
