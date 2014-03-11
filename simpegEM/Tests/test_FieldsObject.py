import unittest
from SimPEG import *
import simpegEM as EM

class FieldsTest(unittest.TestCase):

    def setUp(self):
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0])
        rxList = EM.FDEM.RxListFDEM(XYZ, 'Ex')
        Tx0 = EM.FDEM.TxFDEM(None, 'VMD', 3, rxList)
        Tx1 = EM.FDEM.TxFDEM(None, 'VMD', 3, rxList)
        Tx2 = EM.FDEM.TxFDEM(None, 'VMD', 2, rxList)
        Tx3 = EM.FDEM.TxFDEM(None, 'VMD', 1, rxList)
        txList = [Tx0,Tx1,Tx2,Tx3]
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        survey = EM.FDEM.SurveyFDEM(txList)
        self.F = EM.FDEM.FieldsFDEM(mesh, survey)
        self.Tx0 = Tx0
        self.Tx1 = Tx1

    def test_SetGet(self):
        F = self.F
        for freq in F.survey.freqs:
            e = np.random.rand(F.mesh.nE, F.survey.nTx[freq])
            F[freq, 'e'] = e
            b = np.random.rand(F.mesh.nF, F.survey.nTx[freq])
            F[freq, 'b'] = b
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





if __name__ == '__main__':
    unittest.main()
