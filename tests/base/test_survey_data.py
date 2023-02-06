import unittest
import discretize
import numpy as np
from SimPEG import survey, utils, data

np.random.seed(100)


class TestData(unittest.TestCase):
    def setUp(self):
        mesh = discretize.TensorMesh(
            [np.ones(n) * 5 for n in [10, 11, 12]], [0, 0, -30]
        )
        x = np.linspace(5, 10, 3)
        XYZ = utils.ndgrid(x, x, np.r_[0.0])
        srcLoc = np.r_[0, 0, 0.0]
        receiver_list0 = survey.BaseRx(XYZ)
        Src0 = survey.BaseSrc([receiver_list0], location=srcLoc)
        receiver_list1 = survey.BaseRx(XYZ)
        Src1 = survey.BaseSrc([receiver_list1], location=srcLoc)
        receiver_list2 = survey.BaseRx(XYZ)
        Src2 = survey.BaseSrc([receiver_list2], location=srcLoc)
        receiver_list3 = survey.BaseRx(XYZ)
        Src3 = survey.BaseSrc([receiver_list3], location=srcLoc)
        Src4 = survey.BaseSrc(
            [receiver_list0, receiver_list1, receiver_list2, receiver_list3],
            location=srcLoc,
        )
        source_list = [Src0, Src1, Src2, Src3, Src4]
        mysurvey = survey.BaseSurvey(source_list=source_list)
        self.D = data.Data(mysurvey)

    def test_data(self):
        V = []
        for src in self.D.survey.source_list:
            for rx in src.receiver_list:
                v = np.random.rand(rx.nD)
                V += [v]
                index = self.D.index_dictionary[src][rx]
                self.D.dobs[index] = v
        V = np.concatenate(V)
        self.assertTrue(np.all(V == self.D.dobs))

        D2 = data.Data(self.D.survey, V)
        self.assertTrue(np.all(D2.dobs == self.D.dobs))

    def test_standard_dev(self):
        V = []
        for src in self.D.survey.source_list:
            for rx in src.receiver_list:
                v = np.random.rand(rx.nD)
                V += [v]
                index = self.D.index_dictionary[src][rx]
                self.D.relative_error[index] = v
                self.assertTrue(np.all(v == self.D.relative_error[index]))
        V = np.concatenate(V)
        self.assertTrue(np.all(V == self.D.relative_error))

        D2 = data.Data(self.D.survey, relative_error=V)
        self.assertTrue(np.all(D2.relative_error == self.D.relative_error))

    def test_uniqueSrcs(self):
        srcs = self.D.survey.source_list
        srcs += [srcs[0]]
        self.assertRaises(Exception, survey.BaseSurvey, source_list=srcs)

    def test_sourceIndex(self):
        mysurvey = self.D.survey
        srcs = mysurvey.source_list
        assert mysurvey.get_source_indices([srcs[1], srcs[0]]) == [1, 0]
        assert mysurvey.get_source_indices([srcs[1], srcs[2], srcs[2]]) == [1, 2, 2]
        SrcNotThere = survey.BaseSrc(srcs[0].receiver_list, location=np.r_[0, 0, 0])
        self.assertRaises(KeyError, mysurvey.get_source_indices, [SrcNotThere])
        self.assertRaises(
            KeyError, mysurvey.get_source_indices, [srcs[1], srcs[2], SrcNotThere]
        )


if __name__ == "__main__":
    unittest.main()
