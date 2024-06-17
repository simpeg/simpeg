"""
Tests for resistivity (DC) survey objects.
"""

import pytest

from simpeg.electromagnetics.static.resistivity import Survey
from simpeg.electromagnetics.static.resistivity import sources
from simpeg.electromagnetics.static.resistivity import receivers


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


def test_repr():
    """
    Test the __repr__ method of the survey.
    """
    receivers_list = [
        receivers.Dipole(
            locations_m=[[1, 2, 3], [4, 5, 6]], locations_n=[[7, 8, 9], [10, 11, 12]]
        )
    ]
    sources_list = [
        sources.Dipole(
            receivers_list, location_a=[0.5, 1.5, 2.5], location_b=[4.5, 5.5, 6.5]
        )
    ]
    survey = Survey(source_list=sources_list)
    expected_repr = "Survey(#sources: 1; #data: 2)"
    assert repr(survey) == expected_repr
