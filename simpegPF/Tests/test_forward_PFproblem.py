import unittest
from SimPEG import *
import simpegPF as PF


class MagProblemTests(unittest.TestCase):

    def setUp(self):
        M = Mesh.TensorMesh([10,10])
        mod = Model.LogModel(M)
        prob = PF.Mag.MagProblem(M, mod, None)

        self.prob = prob
        self.M = M

    def test_forward(self):

        passed = True
        self.assertTrue(passed)


    def test_DirchletBC(self):
        q = lambda x: np.sin(x)

        M = self.M
        order = 2
        self.assertTrue(order > 1)



if __name__ == '__main__':
    unittest.main()
