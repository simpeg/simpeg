import pytest
import unittest
import numpy as np
from simpeg import survey, utils, data

from simpeg.survey import BaseRx, BaseSrc, BaseSurvey

np.random.seed(100)


class TestData(unittest.TestCase):
    def setUp(self):
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


class TestSurveySlice:
    """
    Test BaseSurvey's slices for flat arrays.
    """

    def build_receiver(self, n_locs: int):
        locs = np.ones(n_locs)[:, np.newaxis]
        return BaseRx(locs)

    def test_single_source(self):
        """
        Test slicing a survey with a single source.
        """
        n_locs = (4, 7)
        receivers = [self.build_receiver(n_locs=i) for i in n_locs]
        source = BaseSrc(receivers)
        test_survey = BaseSurvey([source])
        assert test_survey.get_slice(source, receivers[0]) == slice(0, 4)
        assert test_survey.get_slice(source, receivers[1]) == slice(4, 4 + 7)

    def test_multiple_sources_shared_receivers(self):
        """
        Test slicing a survey with multiple sources and shared receivers.
        """
        n_locs = (4, 7)
        receivers = [self.build_receiver(n_locs=i) for i in n_locs]
        sources = [BaseSrc(receivers), BaseSrc(receivers)]
        test_survey = BaseSurvey(sources)
        assert test_survey.get_slice(sources[0], receivers[0]) == slice(0, 4)
        assert test_survey.get_slice(sources[0], receivers[1]) == slice(4, 4 + 7)
        assert test_survey.get_slice(sources[1], receivers[0]) == slice(11, 11 + 4)
        assert test_survey.get_slice(sources[1], receivers[1]) == slice(15, 15 + 7)

    def test_multiple_sources(self):
        """
        Test slicing a survey with multiple sources.
        """
        receivers_a = [self.build_receiver(n_locs=i) for i in (4, 7)]
        receivers_b = [self.build_receiver(n_locs=i) for i in (8, 3)]
        sources = [BaseSrc(receivers_a), BaseSrc(receivers_b)]
        test_survey = BaseSurvey(sources)
        assert test_survey.get_slice(sources[0], receivers_a[0]) == slice(0, 4)
        assert test_survey.get_slice(sources[0], receivers_a[1]) == slice(4, 4 + 7)
        assert test_survey.get_slice(sources[1], receivers_b[0]) == slice(11, 11 + 8)
        assert test_survey.get_slice(sources[1], receivers_b[1]) == slice(19, 19 + 3)

    @pytest.mark.parametrize("missing", ["source", "receiver", "both"])
    def test_missing_source_receiver(self, missing):
        """
        Test error on missing source-receiver pair.
        """
        # Generate a survey
        receivers_a = [self.build_receiver(n_locs=i) for i in (4, 7)]
        receivers_b = [self.build_receiver(n_locs=i) for i in (8, 3)]
        sources = [BaseSrc(receivers_a), BaseSrc(receivers_b)]
        test_survey = BaseSurvey(sources)
        # Try to slice with missing source-receiver pair
        src, rx = sources[0], receivers_a[1]
        if missing in ("source", "both"):
            src = BaseSrc()  # new src not in the survey
        if missing in ("receiver", "both"):
            rx = self.build_receiver(1)  # new rx not in the survey
        with pytest.raises(KeyError):
            test_survey.get_slice(src, rx)


if __name__ == "__main__":
    unittest.main()
