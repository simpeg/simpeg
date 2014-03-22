import unittest
from SimPEG import *
import simpegEM as EM

class FieldsTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0.])
        txLoc = np.r_[0,0,0.]
        rxList0 = EM.FDEM.RxListFDEM(XYZ, 'exi,exr,eyi,eyr,ezi,ezr')
        Tx0 = EM.FDEM.TxFDEM(txLoc, 'VMD', 3., rxList0)
        rxList1 = EM.FDEM.RxListFDEM(XYZ, 'bxi,bxr,byi,byr,bzi,bzr')
        Tx1 = EM.FDEM.TxFDEM(txLoc, 'VMD', 3., rxList1)
        rxList2 = EM.FDEM.RxListFDEM(XYZ, 'bxi,eyr')
        Tx2 = EM.FDEM.TxFDEM(txLoc, 'VMD', 2., rxList2)
        rxList3 = EM.FDEM.RxListFDEM(XYZ, 'bxi')
        Tx3 = EM.FDEM.TxFDEM(txLoc, 'VMD', 2., rxList3)
        Tx4 = EM.FDEM.TxFDEM(txLoc, 'VMD', 1., rxList0)
        txList = [Tx0,Tx1,Tx2,Tx3,Tx4]
        survey = EM.FDEM.SurveyFDEM(txList)
        self.F = EM.FDEM.FieldsFDEM(mesh, survey)
        self.D = EM.FDEM.DataFDEM(survey)
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_SetGet(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nTx[freq]
            e = np.random.rand(F.mesh.nE, nFreq)
            F[freq, 'e'] = e
            b = np.random.rand(F.mesh.nF, nFreq)
            F[freq, 'b'] = b
            if nFreq == 1:
                F[freq, 'b'] = Utils.mkvc(b)
            self.assertTrue(np.all(F[freq, 'e'] == e))
            self.assertTrue(np.all(F[freq, 'b'] == b))
            F[freq] = {'b':b,'e':e}
            self.assertTrue(np.all(F[freq, 'e'] == e))
            self.assertTrue(np.all(F[freq, 'b'] == b))

        lastFreq = F[freq]
        self.assertTrue(type(lastFreq) is dict)
        self.assertTrue(sorted([k for k in lastFreq]) == ['b','e'])
        self.assertTrue(np.all(lastFreq['b'] == b))
        self.assertTrue(np.all(lastFreq['e'] == e))

        self.assertTrue(F[3.,'b'].shape == (F.mesh.nF, 2))

        b = np.random.rand(F.mesh.nF, 2)
        F[self.Tx0.freq,'b'] = b
        self.assertTrue(F[self.Tx0]['b'].shape == (F.mesh.nF,))
        self.assertTrue(F[self.Tx0,'b'].shape == (F.mesh.nF,))
        self.assertTrue(np.all(F[self.Tx0,'b'] == b[:,0]))
        self.assertTrue(np.all(F[self.Tx1,'b'] == b[:,1]))

    def test_assertions(self):
        freq = self.F.survey.freqs[0]
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nTx[freq])
        def fun(): self.F[freq, 'b'] = bWrongSize
        self.assertRaises(AssertionError, fun)
        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)
        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)
        def fun(): self.F[freq,'notThere']
        self.assertRaises(KeyError, fun)

    def test_uniqueTxs(self):
        txs = self.F.survey.txList
        txs += [txs[0]]
        self.assertRaises(AssertionError, EM.FDEM.SurveyFDEM, txs)


    def test_dataFDEM(self):
        V = []
        for tx in self.D.survey.txList:
            v = np.random.rand(tx.nD)
            V += [v]
            self.D[tx] = v
            self.assertTrue(np.all(v == self.D[tx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == Utils.mkvc(self.D)))

        D2 = EM.FDEM.DataFDEM(self.D.survey, V)
        self.assertTrue(np.all(Utils.mkvc(D2) == Utils.mkvc(self.D)))

    def test_FieldProjections(self):
        F = self.F
        for freq in F.survey.freqs:
            nFreq = F.survey.nTx[freq]
            e = np.random.rand(F.mesh.nE, nFreq)
            b = np.random.rand(F.mesh.nF, nFreq)
            F[freq] = {'b':b,'e':e}

            Txs = F.survey.getTransmitters(freq)
            for ii, tx in enumerate(Txs):
                dat = tx.projectFields(self.mesh, F)
                self.assertTrue(dat.dtype == float)
                dat = dat.reshape((self.XYZ.shape[0], len(tx.rxList.rxTypes)), order='F')
                for jj, rx in enumerate(tx.rxList.rxTypes):
                    fieldType = tx.rxList._projField(rx)
                    u = {'b':b[:,ii], 'e': e[:,ii]}[fieldType]
                    real_or_imag = tx.rxList._projComp(rx)
                    u = getattr(u, real_or_imag)
                    gloc = tx.rxList._projGLoc(rx)
                    d = self.mesh.getInterpolationMat(self.XYZ, gloc)*u
                    self.assertTrue(np.all(dat[:, jj] == d))



if __name__ == '__main__':
    unittest.main()
