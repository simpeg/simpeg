import unittest
from SimPEG import *
import simpegEM as EM

class FieldsTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0.])
        srcLoc = np.r_[0,0,0.]
        rxList0 = EM.FDEM.RxFDEM(XYZ, 'exi')
        Src0 = EM.FDEM.SrcFDEM_MagDipole(srcLoc, 3., [rxList0])
        rxList1 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Src1 = EM.FDEM.SrcFDEM_MagDipole(srcLoc, 3., [rxList1])
        rxList2 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Src2 = EM.FDEM.SrcFDEM_MagDipole(srcLoc, 2., [rxList2])
        rxList3 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Src3 = EM.FDEM.SrcFDEM_MagDipole(srcLoc, 2., [rxList3])
        Src4 = EM.FDEM.SrcFDEM_MagDipole(srcLoc, 1., [rxList0, rxList1, rxList2, rxList3])
        srcList = [Src0,Src1,Src2,Src3,Src4]
        survey = EM.FDEM.SurveyFDEM(srcList)
        self.F = EM.FDEM.FieldsFDEM(mesh, survey)
        self.Src0 = Src0
        self.Src1 = Src1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_SetGet(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nSrcByFreq[freq]
            Srcs = F.survey.getSources(freq)
            e = np.random.rand(F.mesh.nE, nFreq)
            F[Srcs, 'e'] = e
            b = np.random.rand(F.mesh.nF, nFreq)
            F[Srcs, 'b'] = b
            if nFreq == 1:
                F[Srcs, 'b'] = Utils.mkvc(b)
            if e.shape[1] == 1:
                e, b = Utils.mkvc(e), Utils.mkvc(b)
            self.assertTrue(np.all(F[Srcs, 'e'] == e))
            self.assertTrue(np.all(F[Srcs, 'b'] == b))
            F[Srcs] = {'b':b,'e':e}
            self.assertTrue(np.all(F[Srcs, 'e'] == e))
            self.assertTrue(np.all(F[Srcs, 'b'] == b))

        lastFreq = F[Srcs]
        self.assertTrue(type(lastFreq) is dict)
        self.assertTrue(sorted([k for k in lastFreq]) == ['b','e'])
        self.assertTrue(np.all(lastFreq['b'] == b))
        self.assertTrue(np.all(lastFreq['e'] == e))

        Src_f3 = F.survey.getSources(3.)
        self.assertTrue(F[Src_f3,'b'].shape == (F.mesh.nF, 2))

        b = np.random.rand(F.mesh.nF, 2)
        Src_f0 = F.survey.getSources(self.Src0.freq)
        F[Src_f0,'b'] = b
        self.assertTrue(F[self.Src0]['b'].shape == (F.mesh.nF,))
        self.assertTrue(F[self.Src0,'b'].shape == (F.mesh.nF,))
        self.assertTrue(np.all(F[self.Src0,'b'] == b[:,0]))
        self.assertTrue(np.all(F[self.Src1,'b'] == b[:,1]))

    def test_assertions(self):
        freq = self.F.survey.freqs[0]
        Srcs = self.F.survey.getSources(freq)
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nSrcByFreq[freq])
        def fun(): self.F[Srcs, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)
        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)
        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)
        def fun(): self.F[Srcs,'notThere']
        self.assertRaises(KeyError, fun)

    def test_FieldProjections(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nSrcByFreq[freq]
            Srcs = F.survey.getSources(freq)
            e = np.random.rand(F.mesh.nE, nFreq)
            b = np.random.rand(F.mesh.nF, nFreq)
            F[Srcs] = {'b':b,'e':e}

            Srcs = F.survey.getSources(freq)
            for ii, src in enumerate(Srcs):
                for jj, rx in enumerate(src.rxList):
                    dat = rx.projectFields(src, self.mesh, F)
                    self.assertTrue(dat.dtype == float)
                    fieldType = rx.projField
                    u = {'b':b[:,ii], 'e': e[:,ii]}[fieldType]
                    real_or_imag = rx.projComp
                    u = getattr(u, real_or_imag)
                    gloc = rx.projGLoc
                    d = self.mesh.getInterpolationMat(self.XYZ, gloc)*u
                    self.assertTrue(np.all(dat == d))



if __name__ == '__main__':
    unittest.main()
