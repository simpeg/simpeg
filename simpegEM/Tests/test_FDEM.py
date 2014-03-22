import unittest
from SimPEG import *
import simpegEM as EM

TOL = 1e-10

def getProblem(fdemType):
    cs = 5.
    ncx, ncy, ncz = 2, 2, 2
    npad = 3
    hx = Utils.meshTensors(((npad,cs), (ncx,cs), (npad,cs)))
    hy = Utils.meshTensors(((npad,cs), (ncy,cs), (npad,cs)))
    hz = Utils.meshTensors(((npad,cs), (ncz,cs), (npad,cs)))
    mesh = Mesh.TensorMesh([hx,hy,hz])

    model = Model.LogModel(mesh)

    x = np.linspace(5,10,3)
    XYZ = Utils.ndgrid(x,x,np.r_[0])
    if fdemType == 'e':
        rxList = EM.FDEM.RxFDEM(XYZ, 'bxr')
    elif fdemType == 'b':
        rxList = EM.FDEM.RxFDEM(XYZ, 'exi')
    else:
        raise NotImplementedError()

    Tx0 = EM.FDEM.TxFDEM(np.r_[4.,2.,2.], 'VMD', 1e-2, [rxList])

    x = np.linspace(5,10,3)
    XYZ = Utils.ndgrid(x,x,np.r_[0])
    if fdemType == 'e':
        rxList = EM.FDEM.RxFDEM(XYZ, 'eyi')
    elif fdemType == 'b':
        rxList = EM.FDEM.RxFDEM(XYZ, 'eyr')
    else:
        raise NotImplementedError()

    Tx1 = EM.FDEM.TxFDEM(np.r_[4.,2.,2.], 'VMD', 1e-4, [rxList])

    survey = EM.FDEM.SurveyFDEM([Tx0, Tx1])

    if fdemType == 'e':
        prb = EM.FDEM.ProblemFDEM_e(model)
    elif fdemType == 'b':
        prb = EM.FDEM.ProblemFDEM_b(model)
    else:
        raise NotImplementedError()
    prb.pair(survey)

    return prb

class FDEM_DerivTests_e(unittest.TestCase):

    def setUp(self):
        prb = getProblem('e')
        self.prb = prb
        self.sigma = np.log(np.ones(prb.mesh.nC)*1e-3)
        self.survey = prb.survey

    def test_Jvec(self):
        x0 = self.sigma
        def fun(x):
            return self.survey.dpred(x), lambda x: self.prb.Jvec(x0, x)
        passed = Tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=1e-25)
        self.assertTrue(passed)

    def test_Jtvec_adjointTest(self):
        v = np.random.rand(self.survey.nD)
        w = np.random.rand(self.prb.model.nP)

        m = self.sigma
        u = self.prb.fields(m)
        vJw = v.dot(self.prb.Jvec(m, w, u=u))
        wJtv = w.dot(self.prb.Jtvec(m, v, u=u))
        self.assertTrue(vJw - wJtv < TOL)


class FDEM_DerivTests_b(unittest.TestCase):

    def setUp(self):
        prb = getProblem('b')
        self.prb = prb
        self.sigma = np.log(np.ones(prb.mesh.nC)*1e-3)
        self.survey = prb.survey

    def test_Jvec(self):
        x0 = self.sigma
        def fun(x):
            return self.survey.dpred(x), lambda x: self.prb.Jvec(x0, x)
        passed = Tests.checkDerivative(fun, x0, num=3, plotIt=False, eps=1e-25)
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
