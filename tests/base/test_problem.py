import unittest
from SimPEG import Mesh, Problem
import numpy as np


class TestTimeProblem(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([10, 10])
        self.prob = Problem.BaseTimeProblem(mesh)

    def test_timeProblem_setTimeSteps(self):
        self.prob.timeSteps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        trueTS = np.r_[1e-6, 1e-6, 1e-6, 1e-5, 1e-4, 1e-4]
        self.assertTrue(np.all(trueTS == self.prob.timeSteps))

        self.prob.timeSteps = trueTS
        self.assertTrue(np.all(trueTS == self.prob.timeSteps))

        self.assertTrue(self.prob.nT == 6)

        self.assertTrue(np.all(self.prob.times == np.r_[0, trueTS].cumsum()))


if __name__ == '__main__':
    unittest.main()
