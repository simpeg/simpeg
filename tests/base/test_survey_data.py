import re
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
        self.D = data.Data(mysurvey, dobs=np.ones(mysurvey.nD))

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


class BaseFixtures:
    @pytest.fixture
    def sample_survey(self):
        """Create sample Survey object."""
        x = np.linspace(5, 10, 3)
        coordinates = utils.ndgrid(x, x, np.r_[0.0])
        source_location = np.r_[0, 0, 0.0]
        receivers = [survey.BaseRx(coordinates) for i in range(4)]
        sources = [survey.BaseSrc([rx], location=source_location) for rx in receivers]
        sources.append(survey.BaseSrc(receivers, location=source_location))
        return survey.BaseSurvey(source_list=sources)

    @pytest.fixture
    def sample_data(self, sample_survey):
        """Create sample Data object."""
        dobs = np.random.default_rng(seed=42).uniform(size=sample_survey.nD)
        return data.Data(sample_survey, dobs=dobs)


class TestDataIndexing(BaseFixtures):
    """Test indexing of Data object."""

    def get_source_receiver_pairs(self, survey):
        """Return generator for each source-receiver pair in the survey."""
        source_receiver_pairs = (
            (src, rx) for src in survey.source_list for rx in src.receiver_list
        )
        return source_receiver_pairs

    def test_getitem(self, sample_data):
        """Test the ``Data.__getitem__`` method."""
        dobs = sample_data.dobs.copy()

        # Iterate over source-receiver pairs
        survey_slices = sample_data.survey.get_all_slices()
        for src, rx in self.get_source_receiver_pairs(sample_data.survey):
            # Check if the __getitem__ returns the correct slice of the dobs
            expected = dobs[survey_slices[src, rx]]
            np.testing.assert_allclose(sample_data[src, rx], expected)

    def test_setitem(self, sample_data):
        """Test the ``Data.__setitem__`` method."""
        # Override the dobs array for each source-receiver pair
        dobs_new = []
        rng = np.random.default_rng(seed=42)
        for src, rx in self.get_source_receiver_pairs(sample_data.survey):
            _dobs_new_piece = rng.uniform(size=rx.nD)
            sample_data[src, rx] = _dobs_new_piece
            dobs_new.append(_dobs_new_piece)

        # Check that the dobs in the data matches the new one
        dobs_new = np.hstack(dobs_new)
        np.testing.assert_allclose(dobs_new, sample_data.dobs)

    @pytest.mark.filterwarnings(
        "ignore:The `index_dictionary` property has been deprecated."
    )
    def test_index_dictionary(self, sample_data):
        """Test the ``index_dictionary`` property."""
        dobs = sample_data.dobs.copy()

        # Check indices in index_dictionary for each source-receiver pair
        survey_slices = sample_data.survey.get_all_slices()
        for src, rx in self.get_source_receiver_pairs(sample_data.survey):
            expected_slice_ = survey_slices[src, rx]
            indices = sample_data.index_dictionary[src][rx]
            np.testing.assert_allclose(dobs[indices], dobs[expected_slice_])

    def test_deprecated_index_dictionary(self, sample_data):
        """Test deprecation warning in ``index_dictionary``."""
        source = sample_data.survey.source_list[0]
        receiver = source.receiver_list[0]
        with pytest.warns(
            FutureWarning,
            match=re.escape("The `index_dictionary` property has been deprecated."),
        ):
            sample_data.index_dictionary[source][receiver]


class TestSurveySlice:
    """
    Test BaseSurvey's slices for flat arrays.
    """

    def build_receiver(self, n_locs: int):
        locs = np.ones(n_locs)[:, np.newaxis]
        return BaseRx(locs)

    @pytest.mark.parametrize(
        "all_slices", [True, False], ids=["all_slices", "single_slice"]
    )
    def test_single_source(self, all_slices):
        """
        Test slicing a survey with a single source.
        """
        n_locs = (4, 7)
        receivers = [self.build_receiver(n_locs=i) for i in n_locs]
        source = BaseSrc(receivers)
        test_survey = BaseSurvey([source])
        if all_slices:
            expected = {
                (source, receivers[0]): slice(0, 4),
                (source, receivers[1]): slice(4, 4 + 7),
            }
            slices = test_survey.get_all_slices()
            assert slices == expected
        else:
            assert test_survey.get_slice(source, receivers[0]) == slice(0, 4)
            assert test_survey.get_slice(source, receivers[1]) == slice(4, 4 + 7)

    @pytest.mark.parametrize(
        "all_slices", [True, False], ids=["all_slices", "single_slices"]
    )
    def test_multiple_sources_shared_receivers(self, all_slices):
        """
        Test slicing a survey with multiple sources and shared receivers.
        """
        n_locs = (4, 7)
        receivers = [self.build_receiver(n_locs=i) for i in n_locs]
        sources = [BaseSrc(receivers), BaseSrc(receivers)]
        test_survey = BaseSurvey(sources)
        if all_slices:
            expected = {
                (sources[0], receivers[0]): slice(0, 4),
                (sources[0], receivers[1]): slice(4, 4 + 7),
                (sources[1], receivers[0]): slice(11, 11 + 4),
                (sources[1], receivers[1]): slice(15, 15 + 7),
            }
            slices = test_survey.get_all_slices()
            assert slices == expected
        else:
            assert test_survey.get_slice(sources[0], receivers[0]) == slice(0, 4)
            assert test_survey.get_slice(sources[0], receivers[1]) == slice(4, 4 + 7)
            assert test_survey.get_slice(sources[1], receivers[0]) == slice(11, 11 + 4)
            assert test_survey.get_slice(sources[1], receivers[1]) == slice(15, 15 + 7)

    @pytest.mark.parametrize(
        "all_slices", [True, False], ids=["all_slices", "single_slices"]
    )
    def test_multiple_sources(self, all_slices):
        """
        Test slicing a survey with multiple sources.
        """
        receivers_a = [self.build_receiver(n_locs=i) for i in (4, 7)]
        receivers_b = [self.build_receiver(n_locs=i) for i in (8, 3)]
        srcs = [BaseSrc(receivers_a), BaseSrc(receivers_b)]
        test_survey = BaseSurvey(srcs)
        if all_slices:
            expected = {
                (srcs[0], receivers_a[0]): slice(0, 4),
                (srcs[0], receivers_a[1]): slice(4, 4 + 7),
                (srcs[1], receivers_b[0]): slice(11, 11 + 8),
                (srcs[1], receivers_b[1]): slice(19, 19 + 3),
            }
            slices = test_survey.get_all_slices()
            assert slices == expected
        else:
            assert test_survey.get_slice(srcs[0], receivers_a[0]) == slice(0, 4)
            assert test_survey.get_slice(srcs[0], receivers_a[1]) == slice(4, 4 + 7)
            assert test_survey.get_slice(srcs[1], receivers_b[0]) == slice(11, 11 + 8)
            assert test_survey.get_slice(srcs[1], receivers_b[1]) == slice(19, 19 + 3)

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
        msg = re.escape(
            f"Source '{src}' and receiver '{rx}' pair " "is not part of the survey."
        )
        with pytest.raises(KeyError, match=msg):
            test_survey.get_slice(src, rx)


if __name__ == "__main__":
    unittest.main()
