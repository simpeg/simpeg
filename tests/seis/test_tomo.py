import numpy as np
import scipy.sparse as sp
import unittest

import discretize
from SimPEG.seismic import straight_ray_tomography as tomo
from SimPEG import tests, maps, utils

TOL = 1e-5
FLR = 1e-14


class TomoTest(unittest.TestCase):
    def setUp(self):

        nC = 20
        M = discretize.TensorMesh([nC, nC])
        y = np.linspace(0.0, 1.0, nC // 2)
        rlocs = np.c_[y * 0 + M.vectorCCx[-1], y]
        rx = tomo.Rx(locations=rlocs)

        source_list = [
            tomo.Src(location=np.r_[M.vectorCCx[0], yi], receiver_list=[rx]) for yi in y
        ]

        survey = tomo.Survey(source_list)
        problem = tomo.Simulation(M, survey=survey, slownessMap=maps.IdentityMap(M))

        self.M = M
        self.problem = problem
        self.survey = survey

    def test_deriv(self):
        s = utils.mkvc(utils.model_builder.randomModel(self.M.vnC)) + 1.0

        def fun(x):
            return self.problem.dpred(x), lambda x: self.problem.Jvec(s, x)

        return tests.check_derivative(fun, s, num=4, plotIt=False, eps=FLR)


if __name__ == "__main__":
    unittest.main()
