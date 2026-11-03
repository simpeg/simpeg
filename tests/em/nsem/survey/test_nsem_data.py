import numpy as np
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.natural_source.survey import Data


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
                np.testing.assert_almost_equal(rx.locations_e, [self.loc])
        np.testing.assert_almost_equal(data_obj.dobs, np.array([0.5, 0.0, 0.5, 1.0]))


class TestNSEMDataToRecArray2D:
    """Regression tests for toRecArray with 2D (two-column) locations.

    Covers https://github.com/simpeg/simpeg/issues/1556: toRecArray raised
    a ValueError for surveys whose receiver locations have fewer than
    three columns, because the zero padding was built with a single row.
    """

    @classmethod
    def setup_class(cls):
        cls.freqs = [0.01, 0.05]
        # 2D survey: locations are (profile coordinate, elevation) pairs
        cls.locs = np.c_[np.arange(4.0), np.zeros(4)]
        rx_list = [
            nsem.receivers.Impedance(
                locations_e=cls.locs,
                locations_h=cls.locs,
                orientation="yx",
                component="real",
            ),
            nsem.receivers.Impedance(
                locations_e=cls.locs,
                locations_h=cls.locs,
                orientation="yx",
                component="imag",
            ),
        ]
        src_list = [nsem.sources.Planewave(rx_list, frequency=f) for f in cls.freqs]
        cls.survey = nsem.Survey(src_list)
        cls.dobs = np.arange(cls.survey.nD, dtype=float)
        cls.data_obj = Data(survey=cls.survey, dobs=cls.dobs)

    def test_to_rec_array_2d(self):
        rec = self.data_obj.toRecArray()
        n_loc = len(self.locs)
        assert rec.shape == (n_loc * len(self.freqs),)
        # frequencies repeat per source block
        np.testing.assert_allclose(
            rec["freq"], np.repeat(self.freqs, n_loc)
        )
        # locations are zero-padded to (x, y, z): the two given columns
        # land in y and z, x is zero
        np.testing.assert_allclose(rec["x"], np.zeros(n_loc * len(self.freqs)))
        np.testing.assert_allclose(
            rec["y"], np.tile(self.locs[:, 0], len(self.freqs))
        )
        np.testing.assert_allclose(
            rec["z"], np.tile(self.locs[:, 1], len(self.freqs))
        )
        # impedance values land in the matching real/imaginary columns
        n_per_src = self.survey.nD // len(self.freqs)
        for i in range(len(self.freqs)):
            block = slice(i * n_loc, (i + 1) * n_loc)
            src_dobs = self.dobs[i * n_per_src : (i + 1) * n_per_src]
            np.testing.assert_allclose(rec["zyxr"][block], src_dobs[:n_loc])
            np.testing.assert_allclose(rec["zyxi"][block], src_dobs[n_loc:])

    def test_to_rec_array_2d_complex(self):
        rec = self.data_obj.toRecArray("Complex")
        n_loc = len(self.locs)
        n_per_src = self.survey.nD // len(self.freqs)
        expected = np.concatenate(
            [
                self.dobs[i * n_per_src : i * n_per_src + n_loc]
                + 1j * self.dobs[i * n_per_src + n_loc : (i + 1) * n_per_src]
                for i in range(len(self.freqs))
            ]
        )
        np.testing.assert_allclose(np.asarray(rec["zyx"]).ravel(), expected)
