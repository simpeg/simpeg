import unittest
import numpy as np
import discretize
from SimPEG import simulation, survey, maps


class TestLinearSimulation(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([100])
        self.sim = simulation.ExponentialSinusoidSimulation(
            mesh=mesh, model_map=maps.IdentityMap(mesh),
        )

        mtrue = np.zeros(mesh.nC)
        mtrue[mesh.vectorCCx > 0.3] = 1.0
        mtrue[mesh.vectorCCx > 0.45] = -0.5
        mtrue[mesh.vectorCCx > 0.6] = 0

        self.mtrue = mtrue

    def test_forward(self):
        data = np.r_[
            7.50000000e-02,
            5.34102961e-02,
            5.26315566e-03,
            -3.92235199e-02,
            -4.22361894e-02,
            -1.29419602e-02,
            1.30060891e-02,
            1.73572943e-02,
            7.78056876e-03,
            -1.49689823e-03,
            -4.50212858e-03,
            -3.14559131e-03,
            -9.55761370e-04,
            3.53963158e-04,
            7.24902205e-04,
            6.06022770e-04,
            3.36635644e-04,
            7.48637479e-05,
            -1.10094573e-04,
            -1.84905476e-04,
        ]

        assert np.allclose(data, self.sim.dpred(self.mtrue))

    def test_make_synthetic_data(self):
        dclean = self.sim.dpred(self.mtrue)
        data = self.sim.make_synthetic_data(self.mtrue)
        assert np.all(data.relative_error == 0.05 * np.ones_like(dclean))


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

        self.assertTrue(np.all(self.sim.times == np.r_[0, true_time_steps].cumsum()))

        self.sim.t0 = 1
        self.assertTrue(np.all(self.sim.times == np.r_[1, true_time_steps].cumsum()))


if __name__ == "__main__":
    unittest.main()
