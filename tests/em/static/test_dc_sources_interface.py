"""
Test interface for some DC sources.
"""
import pytest
import numpy as np
from SimPEG.electromagnetics.static import resistivity as dc


class TestDipoleLocations:
    r"""
    Test the location, location_a and location_b arguments for the Dipole

    Considering that `location`, `location_a`, `location_b` can be None or not
    None, then we have 8 different possible combinations.


    .. code::

        | location | location_a | location_b | Result |
        |----------|------------|------------|--------|
        | None     | None       | None       | Error  |
        | None     | None       | not None   | Error  |
        | None     | not None   | None       | Error  |
        | None     | not None   | not None   | Run    |
        | not None | None       | None       | Run    |
        | not None | None       | not None   | Error  |
        | not None | not None   | None       | Error  |
        | not None | not None   | not None   | Error  |
    """

    @pytest.fixture
    def receiver(self):
        """Sample DC dipole receiver."""
        receiver = dc.receivers.Dipole(
            locations_m=np.array([[-100, 0]]),
            locations_n=np.array([[100, 0]]),
            data_type="volt",
        )
        return receiver

    def test_all_nones(self, receiver):
        """
        Test error being raised when passing all location as None
        """
        msg = "Found 'location', 'location_a' and 'location_b' as None. "
        with pytest.raises(TypeError, match=msg):
            dc.sources.Dipole(
                receiver_list=[receiver],
                location_a=None,
                location_b=None,
                location=None,
            )

    @pytest.mark.parametrize("electrode", ("a", "b", "both"))
    def test_not_nones(self, receiver, electrode):
        """
        Test error after location as not None, and location_a and/or location_b
        as not None
        """
        msg = (
            "Found 'location_a' and/or 'location_b' as not None values. "
            "When passing a not None value for 'location', 'location_a' and "
            "'location_b' should be set to None."
        )
        electrode_a = np.array([-1.0, 0.0])
        electrode_b = np.array([1.0, 0.0])
        if electrode == "a":
            kwargs = dict(location_a=electrode_a, location_b=None)
        elif electrode == "b":
            kwargs = dict(location_a=None, location_b=electrode_b)
        else:
            kwargs = dict(location_a=electrode_a, location_b=electrode_b)
        with pytest.raises(TypeError, match=msg):
            dc.sources.Dipole(
                receiver_list=[receiver],
                location=[electrode_a, electrode_b],
                **kwargs,
            )

    @pytest.mark.parametrize("none_electrode", ("a", "b"))
    def test_single_location_as_none(self, receiver, none_electrode):
        """
        Test error after location is None and one of location_a or location_b
        is also None.
        """
        msg = (
            f"Invalid 'location_{none_electrode}' set to None. "
            "When 'location' is None, both 'location_a' and 'location_b' "
            "should be set to a value different than None."
        )
        electrode_a = np.array([-1.0, 0.0])
        electrode_b = np.array([1.0, 0.0])
        if none_electrode == "a":
            kwargs = dict(location_a=None, location_b=electrode_b)
        else:
            kwargs = dict(location_a=electrode_a, location_b=None)
        with pytest.raises(TypeError, match=msg):
            dc.sources.Dipole(
                receiver_list=[receiver],
                location=None,
                **kwargs,
            )

    def test_location_none(self, receiver):
        """
        Test if object is correctly initialized with location set to None
        """
        electrode_a = np.array([-1.0, 0.0])
        electrode_b = np.array([1.0, 0.0])
        source = dc.sources.Dipole(
            receiver_list=[receiver],
            location_a=electrode_a,
            location_b=electrode_b,
            location=None,
        )
        assert isinstance(source.location, np.ndarray)
        assert len(source.location) == 2
        np.testing.assert_allclose(source.location, [electrode_a, electrode_b])

    def test_location_not_none(self, receiver):
        """
        Test if object is correctly initialized with location is set
        """
        electrode_a = np.array([-1.0, 0.0])
        electrode_b = np.array([1.0, 0.0])
        source = dc.sources.Dipole(
            receiver_list=[receiver],
            location=[electrode_a, electrode_b],
        )
        assert isinstance(source.location, np.ndarray)
        assert len(source.location) == 2
        np.testing.assert_allclose(source.location, [electrode_a, electrode_b])
