import numpy as np
import scipy.sparse as sp
import unittest

from SimPEG.SEIS import StraightRay
from SimPEG import Tests, Utils, Mesh, Maps

TOL = 1e-5
FLR = 1e-14


class TomoTest(unittest.TestCase):

    def setUp(self):

        nC = 20
        M = Mesh.TensorMesh([nC, nC])
        y = np.linspace(0., 1., nC/2)
        rlocs = np.c_[y*0+M.vectorCCx[-1], y]
        rx = StraightRay.Rx(rlocs, None)

        srcList = [
            StraightRay.Src(loc=np.r_[M.vectorCCx[0], yi], rxList=[rx])
            for yi in y
        ]

        survey = StraightRay.Survey(srcList)
        problem = StraightRay.Problem(M, slownessMap=Maps.IdentityMap(M))
        problem.pair(survey)

        self.M = M
        self.problem = problem
        self.survey = survey

    def test_deriv(self):
        s = Utils.mkvc(Utils.ModelBuilder.randomModel(self.M.vnC)) + 1.

        def fun(x):
            return self.survey.dpred(x), lambda x: self.problem.Jvec(s, x)
        return Tests.checkDerivative(fun, s, num=4, plotIt=False, eps=FLR)

if __name__ == '__main__':
    unittest.main()



