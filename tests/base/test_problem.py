import unittest
import discretize
import discretize
from SimPEG import maps
from SimPEG import simulation
import numpy as np


class TestTimeSimulation(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh([10, 10])
        self.sim = simulation.BaseTimeSimulation(mesh)

    def test_timeProblem_setTimeSteps(self):
        self.sim.time_steps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        trueTS = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.assertTrue(np.all(trueTS == self.sim.time_steps))

        self.sim.time_steps = trueTS
        self.assertTrue(np.all(trueTS == self.sim.time_steps))

        self.assertTrue(self.sim.nT == 6)

        self.assertTrue(np.all(self.sim.times == np.r_[0, trueTS].cumsum()))


if __name__ == "__main__":
    unittest.main()
