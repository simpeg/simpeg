"""
Test the read_dcip_xyz function.
"""

import pytest
from simpeg.utils.io_utils import read_dcip_xyz

CONTENT_3D = """XA     YA     ZA     XB     YB     ZB     XM     YM     ZM     XN     YN     ZN     data uncertainty
0.8434 0.8218 0.0229 0.8034 0.1207 0.1279 0.4360 0.5560 0.8582 0.9684 0.0521 0.0093 0.1 0.001
0.2081 0.3504 0.8834 0.4551 0.1298 0.4944 0.7215 0.2402 0.8226 0.0799 0.0691 0.6973 0.2 0.002
0.4743 0.9687 0.0158 0.8159 0.2911 0.0481 0.9445 0.5765 0.4865 0.0799 0.2187 0.6148 0.3 0.003
0.3407 0.1419 0.1799 0.0700 0.0299 0.3831 0.4400 0.0388 0.0446 0.0090 0.9237 0.4695 0.4 0.004
0.9619 0.8549 0.7864 0.7285 0.8820 0.4608 0.3499 0.2200 0.2380 0.9353 0.1477 0.2109 0.5 0.005
0.8027 0.9744 0.9024 0.1708 0.5711 0.8938 0.9324 0.9748 0.8475 0.3110 0.7021 0.0794 0.6 0.006
0.3795 0.5080 0.0895 0.3669 0.9556 0.0493 0.1074 0.4350 0.7287 0.1916 0.5331 0.3852 0.7 0.007
0.7773 0.9347 0.4552 0.9284 0.3111 0.1993 0.8203 0.2262 0.0173 0.5669 0.9544 0.8209 0.8 0.008
0.1802 0.3875 0.5838 0.1337 0.9864 0.4771 0.7908 0.8296 0.0046 0.5742 0.6173 0.1196 0.9 0.009
0.3756 0.0674 0.8620 0.6721 0.7010 0.5282 0.5030 0.8407 0.2609 0.0020 0.2154 0.8069 1.0 0.010"""  # noqa: E501

CONTENT_2D = """XA     ZA     XB     ZB     XM     ZM     XN     ZN     data uncertainty
0.8434 0.0229 0.8034 0.1279 0.4360 0.8582 0.9684 0.0093 0.1 0.001
0.2081 0.8834 0.4551 0.4944 0.7215 0.8226 0.0799 0.6973 0.2 0.002
0.4743 0.0158 0.8159 0.0481 0.9445 0.4865 0.0799 0.6148 0.3 0.003
0.3407 0.1799 0.0700 0.3831 0.4400 0.0446 0.0090 0.4695 0.4 0.004
0.9619 0.7864 0.7285 0.4608 0.3499 0.2380 0.9353 0.2109 0.5 0.005
0.8027 0.9024 0.1708 0.8938 0.9324 0.8475 0.3110 0.0794 0.6 0.006
0.3795 0.0895 0.3669 0.0493 0.1074 0.7287 0.1916 0.3852 0.7 0.007
0.7773 0.4552 0.9284 0.1993 0.8203 0.0173 0.5669 0.8209 0.8 0.008
0.1802 0.5838 0.1337 0.4771 0.7908 0.0046 0.5742 0.1196 0.9 0.009
0.3756 0.8620 0.6721 0.5282 0.5030 0.2609 0.0020 0.8069 1.0 0.010"""

VALID_DATA_TYPES = ["volt", "apparent_resistivity", "apparent_chargeability"]

ALL_ELECTRODE_COLS = [
    "XA",
    "YA",
    "ZA",
    "XB",
    "YB",
    "ZB",
    "XM",
    "YM",
    "ZM",
    "XN",
    "YN",
    "ZN",
]  # noqa: E501


@pytest.fixture
def sample_file_3d(tmp_path):
    fname = tmp_path / "dcip_sample.xyz"
    with fname.open(mode="w") as f:
        f.write(CONTENT_3D)
    return fname


@pytest.fixture
def sample_file_2d(tmp_path):
    fname = tmp_path / "dcip_sample.xyz"
    with fname.open(mode="w") as f:
        f.write(CONTENT_2D)
    return fname


class TestNewImplementation:

    return_data_object = False

    @pytest.mark.parametrize("data_type", VALID_DATA_TYPES)
    def test_3d(self, sample_file_3d, data_type):
        """
        Test read_dcip_xyz.
        """
        survey, data_dict = read_dcip_xyz(
            sample_file_3d,
            data_type=data_type,
            return_data_object=self.return_data_object,
        )
        assert "data" in data_dict
        assert "uncertainty" in data_dict
        receivers = [rx for source in survey.source_list for rx in source.receiver_list]
        assert len(survey.source_list) == 10
        assert len(receivers) == 10

    @pytest.mark.parametrize("data_type", VALID_DATA_TYPES)
    def test_2d(self, sample_file_2d, data_type):
        """
        Test read_dcip_xyz.
        """
        survey, data_dict = read_dcip_xyz(
            sample_file_2d,
            data_type=data_type,
            return_data_object=self.return_data_object,
        )
        assert "data" in data_dict
        assert "uncertainty" in data_dict
        receivers = [rx for source in survey.source_list for rx in source.receiver_list]
        assert len(survey.source_list) == 10
        assert len(receivers) == 10

    @pytest.mark.parametrize("remove", ALL_ELECTRODE_COLS)
    def test_invalid_header(self, tmp_path, remove):
        invalid_header = ALL_ELECTRODE_COLS.copy()
        invalid_header.remove(remove)
        # Generate mock file
        fname = tmp_path / "dcip_sample.xyz"
        with fname.open(mode="w") as f:
            f.write(" ".join(invalid_header))

        with pytest.raises(ValueError, match="Invalid file header"):
            read_dcip_xyz(
                fname,
                data_type="volt",
                return_data_object=self.return_data_object,
            )
