import numpy as np
from SimPEG.electromagnetics.natural_source.survey import Data


class TestNSEMData:
    @classmethod
    def setup_class(cls):
        cls.freqs = [0.01, 0.05]
        cls.loc = [48.0, -98.0, 100.0]
        data_types = [
            ("freq", float),
            ("x", float),
            ("y", float),
            ("z", float),
            ("zxxr", float),
            ("zxxi", float),
        ]
        full_array = np.array(
            [
                [[cls.freqs[0], cls.loc[0], cls.loc[1], cls.loc[2], 5.0e-01, 0.0e00]],
                [[cls.freqs[1], cls.loc[0], cls.loc[1], cls.loc[2], 5.0e-01, 1.0e00]],
            ]
        )
        cls.rec_array = np.ma.masked_array(full_array, mask=np.isnan(full_array)).view(
            data_types
        )

    def test_from_rec_array(self):
        """test for class instantiation from a record array"""

        data_obj = Data.fromRecArray(recArray=self.rec_array)
        assert data_obj.survey.frequencies == self.freqs
        assert len(data_obj.survey.source_list) == 2
        for src in data_obj.survey.source_list:
            assert len(src.receiver_list) == 2  # one real, one imaginary component
            for rx in src.receiver_list:
                np.testing.assert_almost_equal(rx.locations, [self.loc])
        np.testing.assert_almost_equal(data_obj.dobs, np.array([0.5, 0.0, 0.5, 1.0]))
