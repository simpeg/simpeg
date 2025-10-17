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
    """Tests after removing the source_type argument and property."""

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


class TestDeprecatedOption:
    """Test the deprecated ``option`` argument in ``drape_electrodes_on_topography``."""

    @pytest.fixture
    def mesh(self):
        return TensorMesh((5, 5, 5))

    def test_warning(self, mesh):
        """Test if warning is raised after passing ``option`` as argument."""
        receivers_list = [
            receivers.Dipole(
                locations_m=[[1, 2, 3], [4, 5, 6]],
                locations_n=[[7, 8, 9], [10, 11, 12]],
            )
        ]
        sources_list = [
            sources.Dipole(
                receivers_list, location_a=[0.5, 1.5, 2.5], location_b=[4.5, 5.5, 6.5]
            )
        ]
        survey = Survey(source_list=sources_list)
        msg = (
            "Argument ``option`` is deprecated in favor of ``topo_cell_cutoff`` "
            "and will be removed in SimPEG v0.27.0."
        )
        active_cells = np.ones(mesh.n_cells, dtype=bool)
        with pytest.warns(FutureWarning, match=msg):
            survey.drape_electrodes_on_topography(mesh, active_cells, option="top")


def test_repr():
    """Test the __repr__ method of the survey."""
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
