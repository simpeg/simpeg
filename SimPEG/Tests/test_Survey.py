import unittest
from SimPEG import *

class DataTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0.])
        txLoc = np.r_[0,0,0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Tx0 = Survey.BaseTx(txLoc, 'VMD', [rxList0])
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Tx1 = Survey.BaseTx(txLoc, 'VMD', [rxList1])
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Tx2 = Survey.BaseTx(txLoc, 'VMD', [rxList2])
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Tx3 = Survey.BaseTx(txLoc, 'VMD', [rxList3])
        Tx4 = Survey.BaseTx(txLoc, 'VMD', [rxList0, rxList1, rxList2, rxList3])
        txList = [Tx0,Tx1,Tx2,Tx3,Tx4]
        survey = Survey.BaseSurvey(txList=txList)
        self.D = Survey.Data(survey)
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_data(self):
        V = []
        for tx in self.D.survey.txList:
            for rx in tx.rxList:
                v = np.random.rand(rx.nD)
                V += [v]
                self.D[tx, rx] = v
                self.assertTrue(np.all(v == self.D[tx, rx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == Utils.mkvc(self.D)))

        D2 = Survey.Data(self.D.survey, V)
        self.assertTrue(np.all(Utils.mkvc(D2) == Utils.mkvc(self.D)))


    def test_uniqueTxs(self):
        txs = self.D.survey.txList
        txs += [txs[0]]
        self.assertRaises(AssertionError, Survey.BaseSurvey, txList=txs)



if __name__ == '__main__':
    unittest.main()
