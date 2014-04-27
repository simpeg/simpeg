import unittest
from SimPEG import *
import simpegEM as EM

class FieldsTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0.])
        txLoc = np.r_[0,0,0.]
        rxList0 = EM.FDEM.RxFDEM(XYZ, 'exi')
        Tx0 = EM.FDEM.TxFDEM(txLoc, 'VMD', 3., [rxList0])
        rxList1 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Tx1 = EM.FDEM.TxFDEM(txLoc, 'VMD', 3., [rxList1])
        rxList2 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Tx2 = EM.FDEM.TxFDEM(txLoc, 'VMD', 2., [rxList2])
        rxList3 = EM.FDEM.RxFDEM(XYZ, 'bxi')
        Tx3 = EM.FDEM.TxFDEM(txLoc, 'VMD', 2., [rxList3])
        Tx4 = EM.FDEM.TxFDEM(txLoc, 'VMD', 1., [rxList0, rxList1, rxList2, rxList3])
        txList = [Tx0,Tx1,Tx2,Tx3,Tx4]
        survey = EM.FDEM.SurveyFDEM(txList)
        self.F = EM.FDEM.FieldsFDEM(mesh, survey)
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_SetGet(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nTxByFreq[freq]
            Txs = F.survey.getTransmitters(freq)
            e = np.random.rand(F.mesh.nE, nFreq)
            F[Txs, 'e'] = e
            b = np.random.rand(F.mesh.nF, nFreq)
            F[Txs, 'b'] = b
            if nFreq == 1:
                F[Txs, 'b'] = Utils.mkvc(b)
            if e.shape[1] == 1:
                e, b = Utils.mkvc(e), Utils.mkvc(b)
            self.assertTrue(np.all(F[Txs, 'e'] == e))
            self.assertTrue(np.all(F[Txs, 'b'] == b))
            F[Txs] = {'b':b,'e':e}
            self.assertTrue(np.all(F[Txs, 'e'] == e))
            self.assertTrue(np.all(F[Txs, 'b'] == b))

        lastFreq = F[Txs]
        self.assertTrue(type(lastFreq) is dict)
        self.assertTrue(sorted([k for k in lastFreq]) == ['b','e'])
        self.assertTrue(np.all(lastFreq['b'] == b))
        self.assertTrue(np.all(lastFreq['e'] == e))

        Tx_f3 = F.survey.getTransmitters(3.)
        self.assertTrue(F[Tx_f3,'b'].shape == (F.mesh.nF, 2))

        b = np.random.rand(F.mesh.nF, 2)
        Tx_f0 = F.survey.getTransmitters(self.Tx0.freq)
        F[Tx_f0,'b'] = b
        self.assertTrue(F[self.Tx0]['b'].shape == (F.mesh.nF,))
        self.assertTrue(F[self.Tx0,'b'].shape == (F.mesh.nF,))
        self.assertTrue(np.all(F[self.Tx0,'b'] == b[:,0]))
        self.assertTrue(np.all(F[self.Tx1,'b'] == b[:,1]))

    def test_assertions(self):
        freq = self.F.survey.freqs[0]
        Txs = self.F.survey.getTransmitters(freq)
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nTxByFreq[freq])
        def fun(): self.F[Txs, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)
        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)
        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)
        def fun(): self.F[Txs,'notThere']
        self.assertRaises(KeyError, fun)

    def test_FieldProjections(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nTxByFreq[freq]
            Txs = F.survey.getTransmitters(freq)
            e = np.random.rand(F.mesh.nE, nFreq)
            b = np.random.rand(F.mesh.nF, nFreq)
            F[Txs] = {'b':b,'e':e}

            Txs = F.survey.getTransmitters(freq)
            for ii, tx in enumerate(Txs):
                for jj, rx in enumerate(tx.rxList):
                    dat = rx.projectFields(tx, self.mesh, F)
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
