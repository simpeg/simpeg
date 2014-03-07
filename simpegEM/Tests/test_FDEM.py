import unittest
from SimPEG import *
import simpegEM as EM

class TDEM_bDerivTests(unittest.TestCase):

    def setUp(self):

        cs = 5.
        ncx = 2
        ncy = 2
        ncz = 2
        npad = 3
        hx = Utils.meshTensors(((npad,cs), (ncx,cs), (npad,cs)))
        hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
        hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
        mesh = Mesh.TensorMesh([hx,hy,hz])

        XY = Utils.ndgrid(np.linspace(20,50,3), np.linspace(20,50,3))
        rxLoc = np.c_[XY, np.ones(XY.shape[0])*40]

        model = Model.LogModel(mesh)

        opts = {'txLoc':0.,
            'txType':'VMD_MVP',
            'rxLoc': rxLoc,
            'rxType':'bz',
            'freq': np.logspace(0,3,4),
            }
        dat = EM.FDEM.SurveyFDEM(**opts)

        prb = EM.FDEM.ProblemFDEM_e(model)
        prb.pair(dat)

        sigma = np.log(np.ones(mesh.nC)*1e-3)

        j_sx = np.zeros(mesh.vnEx)
        j_sx[4,4,4] = 1
        j_s = np.r_[Utils.mkvc(j_sx),np.zeros(mesh.nEy+mesh.nEz)]

        prb.j_s = j_s
        f = prb.fields(sigma)

        self.sigma = sigma
        self.prb = prb
        self.dat = dat

    def test_JVec(self):
        x0 = self.sigma
        def fun(x):
            return self.dat.dpred(x), lambda x: self.prb.Jvec(x0, x)
        passed = Tests.checkDerivative(fun, x0, num=3, plotIt=False)
        self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
