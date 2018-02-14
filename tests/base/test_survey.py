from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
import discretize
from SimPEG import Utils, Simulation, Maps
import SimPEG.NewSurvey as Survey


class TestSurvey(unittest.TestCase):

    def setUp(self):
        x = np.linspace(5, 10, 3)
        XYZ = Utils.ndgrid(x, x, np.r_[0.])
        srcLoc = np.r_[0, 0, 0.]
        rxList0 = Survey.BaseRx(locs=XYZ)
        Src0 = Survey.BaseSrc(rxList=[rxList0], loc=srcLoc)
        rxList1 = Survey.BaseRx(locs=XYZ)
        Src1 = Survey.BaseSrc(rxList=[rxList1], loc=srcLoc)
        rxList2 = Survey.BaseRx(locs=XYZ)
        Src2 = Survey.BaseSrc(rxList=[rxList2], loc=srcLoc)
        rxList3 = Survey.BaseRx(locs=XYZ)
        Src3 = Survey.BaseSrc(rxList=[rxList3], loc=srcLoc)
        Src4 = Survey.BaseSrc(
            rxList=[rxList0, rxList1, rxList2, rxList3], loc=srcLoc
        )
        srcList = [Src0, Src1, Src2, Src3, Src4]
        survey = Survey.BaseSurvey(srcList=srcList)
        self.survey = survey

    def test_uniqueSrcs(self):
        srcs = self.survey.srcList
        srcs += [srcs[0]]
        self.assertRaises(AssertionError, Survey.BaseSurvey, srcList=srcs)

    def test_sourceIndex(self):
        survey = self.survey
        srcs = survey.srcList

        assert survey.getSourceIndex([srcs[1], srcs[0]]) == [1, 0]
        assert survey.getSourceIndex([srcs[1], srcs[2], srcs[2]]) == [1, 2, 2]
        SrcNotThere = Survey.BaseSrc(rxList=srcs[0].rxList, loc=np.r_[0, 0, 0])
        self.assertRaises(KeyError, survey.getSourceIndex, [SrcNotThere])
        self.assertRaises(
            KeyError, survey.getSourceIndex, [srcs[1], srcs[2], SrcNotThere]
        )

    def test_sourceIndex(self):
        survey = self.survey
        src = survey.srcList[-1]
        rxlist = src.rxList

        assert src.getReceiverIndex([rxlist[1], rxlist[0]]) == [1, 0]
        assert src.getReceiverIndex(
            [rxlist[1], rxlist[2], rxlist[2]]
        ) == [1, 2, 2]

        RxNotThere = Survey.BaseRx(locs=np.r_[0, 0, 0])
        self.assertRaises(KeyError, src.getReceiverIndex, [RxNotThere])
        self.assertRaises(
            KeyError, src.getReceiverIndex, [rxlist[1], rxlist[2], RxNotThere]
        )


if __name__ == '__main__':
    unittest.main()
