from __future__ import print_function
import unittest
from SimPEG import Mesh, Problem, Fields, Survey, Utils
import numpy as np
import sys

np.random.seed(32)

if sys.version_info < (3,):
  zero_types = [0, 0.0, np.r_[0], long(0)]
else:
  zero_types = [0, 0.0, np.r_[0]]


class FieldsTest(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10, 11, 12]],
                               [0, 0, -30])
        x = np.linspace(5, 10, 3)
        XYZ = Utils.ndgrid(x, x, np.r_[0.])
        srcLoc = np.r_[0., 0., 0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Src0 = Survey.BaseSrc([rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Src1 = Survey.BaseSrc([rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Src2 = Survey.BaseSrc([rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Src3 = Survey.BaseSrc([rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], loc=srcLoc)
        srcList = [Src0, Src1, Src2, Src3, Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        self.D = Survey.Data(survey)
        self.F = Problem.Fields(mesh, survey, knownFields={'phi': 'CC',
                                                           'e': 'E', 'b': 'F'},
                                dtype={"phi": float, "e": complex,
                                       "b": complex})
        self.Src0 = Src0
        self.Src1 = Src1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nSrc = F.survey.nSrc
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        e = np.random.rand(F.mesh.nE, nSrc)
        F[:, 'e'] = e
        self.assertTrue('b' not in F)
        self.assertTrue('e' in F)

    def test_overlappingFields(self):
        self.assertRaises(AssertionError, Problem.Fields, self.F.mesh,
                          self.F.survey, knownFields={'b': 'F'},
                          aliasFields={'b': ['b', (lambda F, b, ind: b)]})

    def test_SetGet(self):
        F = self.F
        nSrc = F.survey.nSrc
        e = (np.random.rand(F.mesh.nE, nSrc) +
             np.random.rand(F.mesh.nE, nSrc)*1j)
        F[:, 'e'] = e
        b = (np.random.rand(F.mesh.nF, nSrc) +
             np.random.rand(F.mesh.nF, nSrc)*1j)
        F[:, 'b'] = b

        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))
        F[:] = {'b': b, 'e': e}
        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))

        for s in zero_types:
            F[:, 'b'] = s
            self.assertTrue(np.all(F[:, 'b'] == b*0))

        b = np.random.rand(F.mesh.nF, 1)
        print(type(self.Src0), 'here')
        F[self.Src0, 'b'] = b
        self.assertTrue(np.all(F[self.Src0, 'b'] == b))

        b = np.random.rand(F.mesh.nF, 1)
        F[self.Src0, 'b'] = b
        self.assertTrue(np.all(F[self.Src0, 'b'] == b))

        phi = np.random.rand(F.mesh.nC, 2)
        F[[self.Src0, self.Src1], 'phi'] = phi
        self.assertTrue(np.all(F[[self.Src0, self.Src1], 'phi'] == phi))

        fdict = F[:, :]
        self.assertTrue(type(fdict) is dict)
        self.assertTrue(sorted([k for k in fdict]) == ['b', 'e', 'phi'])

        b = np.random.rand(F.mesh.nF, 2)
        F[[self.Src0, self.Src1], 'b'] = b
        self.assertTrue(F[self.Src0]['b'].shape == (F.mesh.nF, 1))
        self.assertTrue(F[self.Src0, 'b'].shape == (F.mesh.nF, 1))
        self.assertTrue(np.all(F[self.Src0, 'b'] == Utils.mkvc(b[:, 0], 2)))
        self.assertTrue(np.all(F[self.Src1, 'b'] == Utils.mkvc(b[:, 1], 2)))

    def test_assertions(self):
        freq = [self.Src0, self.Src1]
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nSrc)

        def fun(): self.F[freq, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)

        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)

        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)

        def fun(): self.F[freq, 'notThere']
        self.assertRaises(KeyError, fun)


class FieldsTest_Alias(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10, 11, 12]],
                               [0, 0, -30])
        x = np.linspace(5, 10, 3)
        XYZ = Utils.ndgrid(x, x, np.r_[0.])
        srcLoc = np.r_[0, 0, 0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Src0 = Survey.BaseSrc([rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Src1 = Survey.BaseSrc([rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Src2 = Survey.BaseSrc([rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Src3 = Survey.BaseSrc([rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], loc=srcLoc)
        srcList = [Src0, Src1, Src2, Src3, Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        self.F = Problem.Fields(mesh, survey, knownFields={'e': 'E'},
                                aliasFields={'b': ['e', 'F',
                                                   (lambda e, ind:
                                                    self.F.mesh.edgeCurl *
                                                    e)]})
        self.Src0 = Src0
        self.Src1 = Src1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nSrc = F.survey.nSrc
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        e = np.random.rand(F.mesh.nE, nSrc)
        F[:, 'e'] = e
        self.assertTrue('b' in F)
        self.assertTrue('e' in F)

    def test_simpleAlias(self):
        F = self.F
        nSrc = F.survey.nSrc
        e = np.random.rand(F.mesh.nE, nSrc)
        F[:, 'e'] = e
        self.assertTrue(np.all(F[:, 'b'] == F.mesh.edgeCurl * e))

        e = np.random.rand(F.mesh.nE, 1)
        F[self.Src0, 'e'] = e
        self.assertTrue(np.all(F[self.Src0, 'b'] == F.mesh.edgeCurl * e))

        def f():
            F[self.Src0, 'b'] = F[self.Src0, 'b']
        self.assertRaises(KeyError, f)  # can't set a alias attr.

    def test_aliasFunction(self):
        def alias(e, ind):
            self.assertTrue(ind[0] is self.Src0)
            return self.F.mesh.edgeCurl * e
        F = Problem.Fields(self.F.mesh, self.F.survey, knownFields={'e': 'E'},
                           aliasFields={'b': ['e', 'F', alias]})
        e = np.random.rand(F.mesh.nE, 1)
        F[self.Src0, 'e'] = e
        F[self.Src0, 'b']

        def alias(e, ind):
            self.assertTrue(type(ind) is list)
            self.assertTrue(ind[0] is self.Src0)
            self.assertTrue(ind[1] is self.Src1)
            return self.F.mesh.edgeCurl * e

        F = Problem.Fields(self.F.mesh, self.F.survey, knownFields={'e': 'E'},
                           aliasFields={'b': ['e', 'F', alias]})
        e = np.random.rand(F.mesh.nE, 2)
        F[[self.Src0, self.Src1], 'e'] = e
        F[[self.Src0, self.Src1], 'b']


class FieldsTest_Time(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10, 11, 12]],
                               [0, 0, -30])
        x = np.linspace(5, 10, 3)
        XYZ = Utils.ndgrid(x, x, np.r_[0.])
        srcLoc = np.r_[0, 0, 0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Src0 = Survey.BaseSrc([rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Src1 = Survey.BaseSrc([rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Src2 = Survey.BaseSrc([rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Src3 = Survey.BaseSrc([rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], loc=srcLoc)
        srcList = [Src0, Src1, Src2, Src3, Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        prob = Problem.BaseTimeProblem(mesh, timeSteps=[(10., 3), (20., 2)])
        survey.pair(prob)
        self.F = Problem.TimeFields(mesh, survey, knownFields={'phi': 'CC',
                                                               'e': 'E',
                                                               'b': 'F'})
        self.Src0 = Src0
        self.Src1 = Src1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nSrc = F.survey.nSrc
        nT = F.survey.prob.nT + 1
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        self.assertTrue('phi' not in F)
        e = np.random.rand(F.mesh.nE, nSrc, nT)
        F[:, 'e', :] = e
        self.assertTrue('e' in F)
        self.assertTrue('b' not in F)
        self.assertTrue('phi' not in F)

    def test_SetGet(self):
        F = self.F
        nSrc = F.survey.nSrc
        nT = F.survey.prob.nT + 1
        e = np.random.rand(F.mesh.nE, nSrc, nT)
        F[:, 'e'] = e
        b = np.random.rand(F.mesh.nF, nSrc, nT)
        F[:, 'b'] = b

        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))
        F[:] = {'b': b, 'e': e}
        self.assertTrue(np.all(F[:, 'e'] == e))
        self.assertTrue(np.all(F[:, 'b'] == b))

        for s in zero_types:
            F[:, 'b'] = s
            self.assertTrue(np.all(F[:, 'b'] == b*0))

        b = np.random.rand(F.mesh.nF, 1, nT)
        F[self.Src0, 'b'] = b
        self.assertTrue(np.all(F[self.Src0, 'b'] == b[:, 0, :]))

        b = np.random.rand(F.mesh.nF, 1, nT)
        F[self.Src0, 'b', 0] = b[:, :, 0]
        self.assertTrue(np.all(F[self.Src0, 'b', 0] == Utils.mkvc(b[:, 0, 0],
                                                                  2)))

        phi = np.random.rand(F.mesh.nC, 2, nT)
        F[[self.Src0, self.Src1], 'phi'] = phi
        self.assertTrue(np.all(F[[self.Src0, self.Src1], 'phi'] == phi))

        fdict = F[:]
        self.assertTrue(type(fdict) is dict)
        self.assertTrue(sorted([k for k in fdict]) == ['b', 'e', 'phi'])

        b = np.random.rand(F.mesh.nF, 2, nT)
        F[[self.Src0, self.Src1], 'b'] = b
        self.assertTrue(F[self.Src0]['b'].shape == (F.mesh.nF, nT))
        self.assertTrue(F[self.Src0, 'b'].shape == (F.mesh.nF, nT))
        self.assertTrue(np.all(F[self.Src0, 'b'] == b[:, 0, :]))
        self.assertTrue(np.all(F[self.Src1, 'b'] == b[:, 1, :]))
        self.assertTrue(np.all(F[self.Src0, 'b', 1] ==
                        Utils.mkvc(b[:, 0, 1], 2)))
        self.assertTrue(np.all(F[self.Src1, 'b', 1] ==
                        Utils.mkvc(b[:, 1, 1], 2)))
        self.assertTrue(np.all(F[self.Src0, 'b', 4] ==
                        Utils.mkvc(b[:, 0, 4], 2)))
        self.assertTrue(np.all(F[self.Src1, 'b', 4] ==
                        Utils.mkvc(b[:, 1, 4], 2)))

        b = np.random.rand(F.mesh.nF, 2, nT)
        F[[self.Src0, self.Src1], 'b', 0] = b[:, :, 0]

    def test_assertions(self):
        freq = [self.Src0, self.Src1]
        bWrongSize = np.random.rand(self.F.mesh.nE, self.F.survey.nSrc)

        def fun(): self.F[freq, 'b'] = bWrongSize
        self.assertRaises(ValueError, fun)

        def fun(): self.F[-999.]
        self.assertRaises(KeyError, fun)

        def fun(): self.F['notRight']
        self.assertRaises(KeyError, fun)

        def fun(): self.F[freq, 'notThere']
        self.assertRaises(KeyError, fun)


class FieldsTest_Time_Aliased(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10, 11, 12]],
                               [0, 0, -30])
        x = np.linspace(5, 10, 3)
        XYZ = Utils.ndgrid(x, x, np.r_[0.])
        srcLoc = np.r_[0, 0, 0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Src0 = Survey.BaseSrc([rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Src1 = Survey.BaseSrc([rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Src2 = Survey.BaseSrc([rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Src3 = Survey.BaseSrc([rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], loc=srcLoc)
        srcList = [Src0, Src1, Src2, Src3, Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        prob = Problem.BaseTimeProblem(mesh, timeSteps=[(10., 3), (20., 2)])
        survey.pair(prob)

        def alias(b, srcInd, timeInd):
            return self.F.mesh.edgeCurl.T * b + timeInd
        self.F = Problem.TimeFields(mesh, survey, knownFields={'b': 'F'},
                                    aliasFields={'e': ['b', 'E', alias]})
        self.Src0 = Src0
        self.Src1 = Src1
        self.mesh = mesh
        self.XYZ = XYZ

    def test_contains(self):
        F = self.F
        nSrc = F.survey.nSrc
        nT = F.survey.prob.nT + 1
        self.assertTrue('b' not in F)
        self.assertTrue('e' not in F)
        b = np.random.rand(F.mesh.nF, nSrc, nT)
        F[:, 'b', :] = b
        self.assertTrue('e' in F)
        self.assertTrue('b' in F)

    def test_simpleAlias(self):
        F = self.F
        nSrc = F.survey.nSrc
        nT = F.survey.prob.nT + 1
        b = np.random.rand(F.mesh.nF, nSrc, nT)
        F[:, 'b', :] = b
        self.assertTrue(np.all(F[:, 'e', 0] == F.mesh.edgeCurl.T * b[:, :, 0]))

        e = list(range(nT))
        for i in range(nT):
            e[i] = F.mesh.edgeCurl.T*b[:, :, i] + i
            e[i] = e[i][:, :, np.newaxis]
        e = np.concatenate(e, axis=2)
        self.assertTrue(np.all(F[:, 'e', :] == e))
        self.assertTrue(np.all(F[self.Src0, 'e', :] == e[:, 0, :]))
        self.assertTrue(np.all(F[self.Src1, 'e', :] == e[:, 1, :]))
        for t in range(nT):
            self.assertTrue(np.all(F[self.Src1, 'e', t] ==
                            Utils.mkvc(e[:, 1, t], 2)))

        b = np.random.rand(F.mesh.nF, nT)
        F[self.Src0, 'b', :] = b
        Cb = F.mesh.edgeCurl.T * b
        for i in range(Cb.shape[1]):
            Cb[:, i] += i
        self.assertTrue(np.all(F[self.Src0, 'e', :] == Cb))

        def f():
            F[self.Src0, 'e'] = F[self.Src0, 'e']
        self.assertRaises(KeyError, f)  # can't set a alias attr.

    def test_aliasFunction(self):
        nT = self.F.survey.prob.nT + 1
        count = [0]

        def alias(e, srcInd, timeInd):
            count[0] += 1
            self.assertTrue(srcInd[0] is self.Src0)
            return self.F.mesh.edgeCurl * e
        F = Problem.TimeFields(self.F.mesh, self.F.survey,
                               knownFields={'e': 'E'},
                               aliasFields={'b': ['e', 'F', alias]})
        e = np.random.rand(F.mesh.nE, 1, nT)
        F[self.Src0, 'e', :] = e
        F[self.Src0, 'b', :]
        # ensure that this is called for every time separately.
        self.assertTrue(count[0] == nT)
        e = np.random.rand(F.mesh.nE, 1, 1)
        F[self.Src0, 'e', 1] = e
        count[0] = 0
        F[self.Src0, 'b', 1]
        self.assertTrue(count[0] == 1)  # ensure that this is called only once.

        def alias(e, srcInd, timeInd):
            count[0] += 1
            self.assertTrue(type(srcInd) is list)
            self.assertTrue(srcInd[0] is self.Src0)
            self.assertTrue(srcInd[1] is self.Src1)
            return self.F.mesh.edgeCurl * e
        F = Problem.TimeFields(self.F.mesh, self.F.survey,
                               knownFields={'e': 'E'},
                               aliasFields={'b': ['e', 'F', alias]})
        e = np.random.rand(F.mesh.nE, 2, nT)
        F[[self.Src0, self.Src1], 'e', :] = e
        count[0] = 0
        F[[self.Src0, self.Src1], 'b', :]

        # ensure that this is called for every time separately.
        self.assertTrue(count[0] == nT)
        e = np.random.rand(F.mesh.nE, 2, 1)
        F[[self.Src0, self.Src1], 'e', 1] = e
        count[0] = 0
        F[[self.Src0, self.Src1], 'b', 1]
        self.assertTrue(count[0] == 1)  # ensure that this is called only once.


if __name__ == '__main__':
    unittest.main()
