from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from SimPEG import *


class TestTimeProblem(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([10,10])
        self.prob = Problem.BaseTimeProblem(mesh)

    def test_timeProblem_setTimeSteps(self):
        self.prob.timeSteps = [(1e-6, 3), 1e-5, (1e-4, 2)]
        trueTS = np.r_[1e-6,1e-6,1e-6,1e-5,1e-4,1e-4]
        self.assertTrue(np.all(trueTS == self.prob.timeSteps))

        self.prob.timeSteps = trueTS
        self.assertTrue(np.all(trueTS == self.prob.timeSteps))

        self.assertTrue(self.prob.nT == 6)

        self.assertTrue(np.all(self.prob.times == np.r_[0,trueTS].cumsum()))


if __name__ == '__main__':
    unittest.main()
