"""
Tests for Spectral IP (SIP) survey objects.
"""

import pytest

from simpeg.electromagnetics.static.spectral_induced_polarization import Survey


class TestRemovedSourceType:
    """
    Tests after removing the source_type argument and property.
    """

    def test_warning_after_argument(self):
        """
        Test warning after passing source_type as argument to the constructor.
        """
        msg = "Argument 'survey_type' is ignored and will be removed in future"
        with pytest.warns(FutureWarning, match=msg):
            survey = Survey(source_list=[], survey_type="dipole-dipole")
        # Check if the object doesn't have a `_survey_type` attribute
        assert not hasattr(survey, "_survey_type")

    def test_warning_removed_property(self):
        """
        Test if warning is raised when accessing the survey_type property.
        """
        survey = Survey(source_list=[])
        msg = "Property 'survey_type' has been removed."
        with pytest.warns(FutureWarning, match=msg):
            survey.survey_type
        with pytest.warns(FutureWarning, match=msg):
            survey.survey_type = "dipole-dipole"
