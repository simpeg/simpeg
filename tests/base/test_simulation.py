from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import discretize
from SimPEG import Utils, Simulation
from SimPEG.NewSurvey import LinearSurvey


class TestSimulation(unittest.TestCase):

    def setUp(self):
        mesh = discretize.TensorMesh([10, 10])
        self.sim = Simulation.BaseTimeSimulation(mesh=mesh)

    def test_time_simulation_time_steps(self):
        self.sim.time_steps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        true_time_steps = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.assertTrue(np.all(true_time_steps == self.sim.time_steps))

if __name__ == '__main__':
    unittest.main()
