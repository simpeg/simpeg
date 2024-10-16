"""
Tests for resistivity (DC) survey objects.
"""

import pytest
import numpy as np

from discretize import TensorMesh
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


class TestDeprecatedIndActive:
    """
    Test the deprecated ``ind_active`` argument in ``drape_electrodes_on_topography``.
    """

    @pytest.fixture
    def mesh(self):
        return TensorMesh((5, 5, 5))

    def test_error(self, mesh):
        """
        Test if error is raised after passing ``ind_active`` as argument.
        """
        survey = Survey(source_list=[])
        msg = "'ind_active' has been deprecated and will be removed in "
        active_cells = np.ones(mesh.n_cells, dtype=bool)
        with pytest.raises(TypeError, match=msg):
            survey.drape_electrodes_on_topography(
                mesh, active_cells, ind_active=active_cells
            )


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
