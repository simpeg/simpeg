import unittest
from SimPEG import *

class DataAndFieldsTest(unittest.TestCase):

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
        self.F = Problem.Fields(mesh, survey, knownFields={'phi':'CC','e':'E','b':'F'}, dtype={"phi":float,"e":complex,"b":complex})
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_overlappingFields(self):
        self.assertRaises(AssertionError, Problem.Fields, self.F.mesh, self.F.survey,
                            knownFields={'b':'F'},
                            aliasFields={'b':['b',(lambda F, b, ind: b)]})

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

    def test_contains(self):
        F = self.F
        nTx = F.survey.nTx
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        e = np.random.rand(F.mesh.nE, nTx)
        F[:, 'e'] = e
        self.assertTrue('b' not in F)
        self.assertTrue('e' in F)

    def test_uniqueTxs(self):
        txs = self.D.survey.txList
        txs += [txs[0]]
        self.assertRaises(AssertionError, Survey.BaseSurvey, txList=txs)

    def test_SetGet(self):
        F = self.F
        nTx = F.survey.nTx
        e = np.random.rand(F.mesh.nE, nTx) + np.random.rand(F.mesh.nE, nTx)*1j
        F[:, 'e'] = e
        b = np.random.rand(F.mesh.nF, nTx) + np.random.rand(F.mesh.nF, nTx)*1j
        F[:, 'b'] = b

        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))
        F[:] = {'b':b,'e':e}
        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))

        for s in [0,0.0,np.r_[0],long(0)]:
            F[:, 'b'] = s
            self.assertTrue(np.all(F[:, 'b'] == b*0))

        b = np.random.rand(F.mesh.nF,1)
        F[self.Tx0, 'b'] = b
        self.assertTrue(np.all(F[self.Tx0, 'b'] == Utils.mkvc(b)))

        b = np.random.rand(F.mesh.nF)
        F[self.Tx0, 'b'] = b
        self.assertTrue(np.all(F[self.Tx0, 'b'] == b))

        phi = np.random.rand(F.mesh.nC,2)
        F[[self.Tx0,self.Tx1], 'phi'] = phi
        self.assertTrue(np.all(F[[self.Tx0,self.Tx1], 'phi'] == phi))

        fdict = F[:,:]
        self.assertTrue(type(fdict) is dict)
        self.assertTrue(sorted([k for k in fdict]) == ['b','e','phi'])

        b = np.random.rand(F.mesh.nF, 2)
        F[[self.Tx0, self.Tx1],'b'] = b
        self.assertTrue(F[self.Tx0]['b'].shape == (F.mesh.nF,))
        self.assertTrue(F[self.Tx0,'b'].shape == (F.mesh.nF,))
        self.assertTrue(np.all(F[self.Tx0,'b'] == b[:,0]))
        self.assertTrue(np.all(F[self.Tx1,'b'] == b[:,1]))

    def test_assertions(self):
        freq = [self.Tx0, self.Tx1]
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nTx)
        def fun(): self.F[freq, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)
        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)
        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)
        def fun(): self.F[freq,'notThere']
        self.assertRaises(KeyError, fun)


class FieldsTest_Alias(unittest.TestCase):

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
        self.F = Problem.Fields(mesh, survey, knownFields={'e':'E'}, aliasFields={'b':['e','F',(lambda e, ind: self.F.mesh.edgeCurl * e)]})
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nTx = F.survey.nTx
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        e = np.random.rand(F.mesh.nE, nTx)
        F[:, 'e'] = e
        self.assertTrue('b' in F)
        self.assertTrue('e' in F)

    def test_simpleAlias(self):
        F = self.F
        nTx = F.survey.nTx
        e = np.random.rand(F.mesh.nE, nTx)
        F[:, 'e'] = e
        self.assertTrue(np.all(F[:, 'b'] == F.mesh.edgeCurl * e ))

        e = np.random.rand(F.mesh.nE,1)
        F[self.Tx0, 'e'] = e
        self.assertTrue(np.all(F[self.Tx0, 'b'] == F.mesh.edgeCurl * Utils.mkvc(e)))

        def f():
            F[self.Tx0, 'b'] = F[self.Tx0, 'b']
        self.assertRaises(KeyError, f) # can't set a alias attr.

    def test_aliasFunction(self):
        def alias(e, ind):
            self.assertTrue(ind is self.Tx0)
            return self.F.mesh.edgeCurl * e
        F = Problem.Fields(self.F.mesh, self.F.survey, knownFields={'e':'E'}, aliasFields={'b':['e','F',alias]})
        e = np.random.rand(F.mesh.nE,1)
        F[self.Tx0, 'e'] = e
        F[self.Tx0, 'b']


        def alias(e, ind):
            self.assertTrue(type(ind) is list)
            self.assertTrue(ind[0] is self.Tx0)
            self.assertTrue(ind[1] is self.Tx1)
            return self.F.mesh.edgeCurl * e
        F = Problem.Fields(self.F.mesh, self.F.survey, knownFields={'e':'E'}, aliasFields={'b':['e','F',alias]})
        e = np.random.rand(F.mesh.nE,2)
        F[[self.Tx0, self.Tx1], 'e'] = e
        F[[self.Tx0, self.Tx1], 'b']


class FieldsTest_Time(unittest.TestCase):

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
        prob = Problem.BaseTimeProblem(mesh, timeSteps=[(10.,3), (20.,2)])
        survey.pair(prob)
        self.F = Problem.TimeFields(mesh, survey, knownFields={'phi':'CC','e':'E','b':'F'})
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nTx = F.survey.nTx
        nT = F.survey.prob.nT + 1
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        self.assertTrue('phi' not in F)
        e = np.random.rand(F.mesh.nE, nTx, nT)
        F[:, 'e', :] = e
        self.assertTrue('e' in F)
        self.assertTrue('b' not in F)
        self.assertTrue('phi' not in F)

    def test_SetGet(self):
        F = self.F
        nTx = F.survey.nTx
        nT = F.survey.prob.nT + 1
        e = np.random.rand(F.mesh.nE, nTx, nT)
        F[:, 'e'] = e
        b = np.random.rand(F.mesh.nF, nTx, nT)
        F[:, 'b'] = b

        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))
        F[:] = {'b':b,'e':e}
        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))

        for s in [0,0.0,np.r_[0],long(0)]:
            F[:, 'b'] = s
            self.assertTrue(np.all(F[:, 'b'] == b*0))

        b = np.random.rand(F.mesh.nF,1,nT)
        F[self.Tx0, 'b'] = b
        self.assertTrue(np.all(F[self.Tx0, 'b'] == b[:,0,:]))

        b = np.random.rand(F.mesh.nF,1,nT)
        F[self.Tx0, 'b', 0] = b[:,:,0]
        self.assertTrue(np.all(F[self.Tx0, 'b', 0] == b[:,0,0]))

        phi = np.random.rand(F.mesh.nC,2,nT)
        F[[self.Tx0,self.Tx1], 'phi'] = phi
        self.assertTrue(np.all(F[[self.Tx0,self.Tx1], 'phi'] == phi))

        fdict = F[:]
        self.assertTrue(type(fdict) is dict)
        self.assertTrue(sorted([k for k in fdict]) == ['b','e','phi'])

        b = np.random.rand(F.mesh.nF, 2, nT)
        F[[self.Tx0, self.Tx1],'b'] = b
        self.assertTrue(F[self.Tx0]['b'].shape == (F.mesh.nF,nT))
        self.assertTrue(F[self.Tx0,'b'].shape == (F.mesh.nF,nT))
        self.assertTrue(np.all(F[self.Tx0,'b'] == b[:,0,:]))
        self.assertTrue(np.all(F[self.Tx1,'b'] == b[:,1,:]))
        self.assertTrue(np.all(F[self.Tx0,'b',1] == b[:,0,1]))
        self.assertTrue(np.all(F[self.Tx1,'b',1] == b[:,1,1]))
        self.assertTrue(np.all(F[self.Tx0,'b',4] == b[:,0,4]))
        self.assertTrue(np.all(F[self.Tx1,'b',4] == b[:,1,4]))


        b = np.random.rand(F.mesh.nF, 2, nT)
        F[[self.Tx0, self.Tx1],'b', 0] = b[:,:,0]

    def test_assertions(self):
        freq = [self.Tx0, self.Tx1]
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nTx)
        def fun(): self.F[freq, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)
        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)
        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)
        def fun(): self.F[freq,'notThere']
        self.assertRaises(KeyError, fun)


class FieldsTest_Time_Aliased(unittest.TestCase):

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
        prob = Problem.BaseTimeProblem(mesh, timeSteps=[(10.,3), (20.,2)])
        survey.pair(prob)
        def alias(b, txInd, timeInd):
            return self.F.mesh.edgeCurl.T * b + timeInd
        self.F = Problem.TimeFields(mesh, survey, knownFields={'b':'F'}, aliasFields={'e':['b','E',alias]})
        self.Tx0 = Tx0
        self.Tx1 = Tx1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nTx = F.survey.nTx
        nT = F.survey.prob.nT + 1
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        b = np.random.rand(F.mesh.nF, nTx, nT)
        F[:, 'b', :] = b
        self.assertTrue('e' in F)
        self.assertTrue('b' in F)


    def test_simpleAlias(self):
        F = self.F
        nTx = F.survey.nTx
        nT = F.survey.prob.nT + 1
        b = np.random.rand(F.mesh.nF, nTx, nT)
        F[:, 'b', :] = b
        self.assertTrue(np.all(F[:, 'e', 0] == F.mesh.edgeCurl.T * b[:,:,0] ))

        e = range(nT)
        for i in range(nT):
            e[i] = F.mesh.edgeCurl.T*b[:,:,i] + i
            e[i] = e[i][:,:,np.newaxis]
        e = np.concatenate(e, axis=2)
        self.assertTrue(np.all(F[:, 'e', :] == e ))
        self.assertTrue(np.all(F[self.Tx0, 'e', :] == e[:,0,:] ))
        self.assertTrue(np.all(F[self.Tx1, 'e', :] == e[:,1,:] ))
        for t in range(nT):
            self.assertTrue(np.all(F[self.Tx1, 'e', t] == e[:,1,t] ))

        b = np.random.rand(F.mesh.nF,nT)
        F[self.Tx0, 'b',:] = b
        Cb = F.mesh.edgeCurl.T * b
        for i in range(Cb.shape[1]):
            Cb[:,i] += i
        self.assertTrue(np.all(F[self.Tx0, 'e',:] == Cb))

        def f():
            F[self.Tx0, 'e'] = F[self.Tx0, 'e']
        self.assertRaises(KeyError, f) # can't set a alias attr.

    def test_aliasFunction(self):
        nT = self.F.survey.prob.nT + 1
        count = [0]
        def alias(e, txInd, timeInd):
            count[0] += 1
            self.assertTrue(txInd is self.Tx0)
            return self.F.mesh.edgeCurl * e
        F = Problem.TimeFields(self.F.mesh, self.F.survey, knownFields={'e':'E'}, aliasFields={'b':['e','F',alias]})
        e = np.random.rand(F.mesh.nE,1,nT)
        F[self.Tx0, 'e', :] = e
        F[self.Tx0, 'b', :]
        self.assertTrue(count[0] == nT) # ensure that this is called for every time separately.
        e = np.random.rand(F.mesh.nE,1,1)
        F[self.Tx0, 'e', 1] = e
        count[0] = 0
        F[self.Tx0, 'b', 1]
        self.assertTrue(count[0] == 1) # ensure that this is called only once.


        def alias(e, txInd, timeInd):
            count[0] += 1
            self.assertTrue(type(txInd) is list)
            self.assertTrue(txInd[0] is self.Tx0)
            self.assertTrue(txInd[1] is self.Tx1)
            return self.F.mesh.edgeCurl * e
        F = Problem.TimeFields(self.F.mesh, self.F.survey, knownFields={'e':'E'}, aliasFields={'b':['e','F',alias]})
        e = np.random.rand(F.mesh.nE,2, nT)
        F[[self.Tx0, self.Tx1], 'e', :] = e
        count[0] = 0
        F[[self.Tx0, self.Tx1], 'b', :]
        self.assertTrue(count[0] == nT) # ensure that this is called for every time separately.
        e = np.random.rand(F.mesh.nE,2, 1)
        F[[self.Tx0, self.Tx1], 'e', 1] = e
        count[0] = 0
        F[[self.Tx0, self.Tx1], 'b', 1]
        self.assertTrue(count[0] == 1) # ensure that this is called only once.


if __name__ == '__main__':
    unittest.main()
