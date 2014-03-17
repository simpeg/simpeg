import unittest
from SimPEG import *
import simpegEM as EM

TOL = 1e-10

class FDEM_bDerivTests(unittest.TestCase):

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

        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0])
        rxList = EM.FDEM.RxListFDEM(XYZ, 'Ex')
        Tx0 = EM.FDEM.TxFDEM(None, 'VMD', 3, rxList)

        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0])
        rxList = EM.FDEM.RxListFDEM(XYZ, 'Ey')
        Tx1 = EM.FDEM.TxFDEM(None, 'VMD', 3, rxList)

        survey = EM.FDEM.SurveyFDEM([Tx0, Tx1])

        prb = EM.FDEM.ProblemFDEM_e(model)
        prb.pair(survey)

        sigma = np.log(np.ones(mesh.nC)*1e-3)

        j_sx = np.zeros(mesh.vnEx)
        j_sx[4,4,4] = 1
        j_s = np.r_[Utils.mkvc(j_sx),np.zeros(mesh.nEy+mesh.nEz)]

        prb.j_s = np.c_[j_s, j_s]
        f = prb.fields(sigma)

        self.sigma = sigma
        self.prb = prb
        self.survey = survey

    def test_Jvec(self):
        x0 = self.sigma
        def fun(x):
            return self.survey.dpred(x), lambda x: self.prb.Jvec(x0, x)
        passed = Tests.checkDerivative(fun, x0, num=3, plotIt=False)
        self.assertTrue(passed)

    def test_Jtvec_adjointTest(self):
        v = np.random.rand(self.survey.nD)
        w = np.random.rand(self.prb.model.nP)

        m = self.sigma
        u = self.prb.fields(m)
        vJw = v.dot(self.prb.Jvec(m, w, u=u))
        wJtv = w.dot(self.prb.Jtvec(m, v, u=u))
        self.assertTrue(vJw - wJtv < TOL)


if __name__ == '__main__':
    unittest.main()
