import unittest
import numpy as np
import discretize
from SimPEG import simulation, survey, maps

class TestLinearSimulation(unittest.TestCase):

    def setUp(self):
        mesh = discretize.TensorMesh([100])
        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh,
            model_map=maps.IdentityMap(mesh),
        )

        mtrue = np.zeros(mesh.nC)
        mtrue[mesh.vectorCCx > 0.3] = 1.
        mtrue[mesh.vectorCCx > 0.45] = -0.5
        mtrue[mesh.vectorCCx > 0.6] = 0

        self.mtrue = mtrue

    def test_forward(self):
        data = np.r_[
            7.04018834e+00, 4.14041969e+00,  -9.93782975e-01,
            -4.46393702e+00, -3.66021155e+00,  -5.42051412e-01,
            1.58193136e+00,   1.60537971e+00, 5.73664455e-01,
            -2.44612295e-01,  -4.48122888e-01,  -2.84138871e-01,
            -7.46161437e-02,   4.26646164e-02,   7.27612633e-02,
            5.90210173e-02, 3.22579286e-02,   6.71960934e-03,
            -1.12292528e-02,  -1.84905476e-02,
        ]

        assert np.allclose(data, self.sim.dpred(self.mtrue))

    def test_make_synthetic_data(self):
        dclean = self.sim.dpred(self.mtrue)
        data = self.sim.make_synthetic_data(self.mtrue)
        assert np.all(
            data.standard_deviation == 0.05 * np.ones_like(dclean)
        )


class TestTimeSimulation(unittest.TestCase):

    def setUp(self):
        mesh = discretize.TensorMesh([10, 10])
        self.sim = simulation.BaseTimeSimulation(mesh=mesh)

    def test_time_simulation_time_steps(self):
        self.sim.time_steps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        true_time_steps = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.assertTrue(np.all(true_time_steps == self.sim.time_steps))

        true_time_steps = np.r_[1e-7, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.sim.time_steps = true_time_steps
        self.assertTrue(np.all(true_time_steps == self.sim.time_steps))

        self.assertTrue(self.sim.nT == 6)

        self.assertTrue(
            np.all(self.sim.times == np.r_[0, true_time_steps].cumsum())
        )

        self.sim.t0 = 1
        self.assertTrue(
            np.all(self.sim.times == np.r_[1, true_time_steps].cumsum())
        )


if __name__ == '__main__':
    unittest.main()
