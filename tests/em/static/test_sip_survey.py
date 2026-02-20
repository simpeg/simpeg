"""
Tests for Spectral IP (SIP) survey objects.
"""

import pytest

from simpeg.electromagnetics.static.spectral_induced_polarization import Survey


class TestRemovedSourceType:
    """
    Tests after removing the source_type argument and property.
    """

    def test_error_after_argument(self):
        """
        Test error after passing ``source_type`` as argument to the constructor.
        """
        msg = "Argument 'survey_type' has been removed"
        with pytest.raises(TypeError, match=msg):
            Survey(source_list=[], survey_type="dipole-dipole")

    def test_error_removed_property(self):
        """
        Test if error is raised when accessing the ``survey_type`` property.
        """
        survey = Survey(source_list=[])
        msg = "'survey_type' has been removed."
        with pytest.raises(AttributeError, match=msg):
            survey.survey_type
        with pytest.raises(AttributeError, match=msg):
            survey.survey_type = "dipole-dipole"
