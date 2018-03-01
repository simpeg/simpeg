from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from SimPEG import Mesh, Survey, Utils

np.random.seed(100)


class TestData(unittest.TestCase):

    def setUp(self):
        mesh = Mesh.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = Utils.ndgrid(x,x,np.r_[0.])
        srcLoc = np.r_[0,0,0.]
        rxList0 = Survey.BaseRx(XYZ, 'exi')
        Src0 = Survey.BaseSrc([rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(XYZ, 'bxi')
        Src1 = Survey.BaseSrc([rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(XYZ, 'bxi')
        Src2 = Survey.BaseSrc([rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(XYZ, 'bxi')
        Src3 = Survey.BaseSrc([rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], loc=srcLoc)
        srcList = [Src0,Src1,Src2,Src3,Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        self.D = Survey.Data(survey)

    def test_data(self):
        V = []
        for src in self.D.survey.srcList:
            for rx in src.rxList:
                v = np.random.rand(rx.nD)
                V += [v]
                self.D[src, rx] = v
                self.assertTrue(np.all(v == self.D[src, rx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == Utils.mkvc(self.D)))

        D2 = Survey.Data(self.D.survey, V)
        self.assertTrue(np.all(Utils.mkvc(D2) == Utils.mkvc(self.D)))

    def test_standard_dev(self):
        V = []
        for src in self.D.survey.srcList:
            for rx in src.rxList:
                v = np.random.rand(rx.nD)
                V += [v]
                self.D.standard_deviation[src, rx] = v
                self.assertTrue(np.all(v == self.D.standard_deviation[src, rx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == Utils.mkvc(self.D.standard_deviation)))

        D2 = Survey.Data(self.D.survey, standard_deviation=V)
        self.assertTrue(np.all(Utils.mkvc(D2.standard_deviation) == Utils.mkvc(self.D.standard_deviation)))

    def test_uniqueSrcs(self):
        srcs = self.D.survey.srcList
        srcs += [srcs[0]]
        self.assertRaises(AssertionError, Survey.BaseSurvey, srcList=srcs)

    def test_sourceIndex(self):
        survey = self.D.survey
        srcs = survey.srcList
        assert survey.getSourceIndex([srcs[1],srcs[0]]) == [1,0]
        assert survey.getSourceIndex([srcs[1],srcs[2],srcs[2]]) == [1,2,2]
        SrcNotThere = Survey.BaseSrc(srcs[0].rxList, loc=np.r_[0,0,0])
        self.assertRaises(KeyError, survey.getSourceIndex, [SrcNotThere])
        self.assertRaises(KeyError, survey.getSourceIndex, [srcs[1],srcs[2],SrcNotThere])

if __name__ == '__main__':
    unittest.main()
