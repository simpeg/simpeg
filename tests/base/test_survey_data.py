from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import discretize
import numpy as np
from SimPEG import survey, utils, data

np.random.seed(100)


class TestData(unittest.TestCase):

    def setUp(self):
        mesh = discretize.TensorMesh([np.ones(n)*5 for n in [10,11,12]],[0,0,-30])
        x = np.linspace(5,10,3)
        XYZ = utils.ndgrid(x,x,np.r_[0.])
        srcLoc = np.r_[0,0,0.]
        rxList0 = survey.BaseRx(XYZ)
        Src0 = survey.BaseSrc([rxList0], location=srcLoc)
        rxList1 = survey.BaseRx(XYZ)
        Src1 = survey.BaseSrc([rxList1], location=srcLoc)
        rxList2 = survey.BaseRx(XYZ)
        Src2 = survey.BaseSrc([rxList2], location=srcLoc)
        rxList3 = survey.BaseRx(XYZ)
        Src3 = survey.BaseSrc([rxList3], location=srcLoc)
        Src4 = survey.BaseSrc([rxList0, rxList1, rxList2, rxList3], location=srcLoc)
        srcList = [Src0,Src1,Src2,Src3,Src4]
        mysurvey = survey.BaseSurvey(source_list=srcList)
        self.D = data.Data(mysurvey)

    def test_data(self):
        V = []
        for src in self.D.survey.source_list:
            for rx in src.receiver_list:
                v = np.random.rand(rx.nD)
                V += [v]
                self.D[src, rx] = v
                self.assertTrue(np.all(v == self.D[src, rx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == utils.mkvc(self.D))) # TODO: think about this

        D2 = data.Data(self.D.survey, V)
        self.assertTrue(np.all(utils.mkvc(D2) == utils.mkvc(self.D)))

    def test_standard_dev(self):
        V = []
        for src in self.D.survey.source_list:
            for rx in src.receiver_list:
                v = np.random.rand(rx.nD)
                V += [v]
                self.D._standard_deviation[src, rx] = v
                self.assertTrue(np.all(v == self.D._standard_deviation[src, rx]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == self.D.standard_deviation))

        D2 = data.Data(self.D.survey, standard_deviation=V)
        self.assertTrue(np.all(D2.standard_deviation == self.D.standard_deviation))

    def test_uniqueSrcs(self):
        srcs = self.D.survey.source_list
        srcs += [srcs[0]]
        self.assertRaises(AssertionError, survey.BaseSurvey, source_list=srcs)

    def test_sourceIndex(self):
        mysurvey = self.D.survey
        srcs = mysurvey.source_list
        assert mysurvey.getSourceIndex([srcs[1],srcs[0]]) == [1,0]
        assert mysurvey.getSourceIndex([srcs[1],srcs[2],srcs[2]]) == [1,2,2]
        SrcNotThere = survey.BaseSrc(srcs[0].receiver_list, location=np.r_[0,0,0])
        self.assertRaises(KeyError, mysurvey.getSourceIndex, [SrcNotThere])
        self.assertRaises(KeyError, mysurvey.getSourceIndex, [srcs[1],srcs[2],SrcNotThere])

if __name__ == '__main__':
    unittest.main()
