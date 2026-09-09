"""
Test imports of static_utils from resisitivty and induced_polarization modules
"""

import pytest

IMPORTS = (
    "electrode_separations",
    "pseudo_locations",
    "apparent_resistivity_from_voltage",
    "plot_pseudosection",
    "generate_dcip_survey",
    "generate_dcip_sources_line",
    "generate_survey_from_abmn_locations",
    "geometric_factor",
    "convert_survey_3d_to_2d_lines",
    "xy_2_lineID",
    "r_unit",
    "gettopoCC",
    "drapeTopotoLoc",
    "genTopography",
    "closestPointsGrid",
    "gen_3d_survey_from_2d_lines",
    "plot_1d_layer_model",
)


@pytest.mark.parametrize("imported_object", IMPORTS)
def test_resisitivity(imported_object):
    """
    Test imports from the resistivity submodule
    """
    import simpeg.electromagnetics.static.resistivity.utils as utils

    assert hasattr(utils, imported_object)


@pytest.mark.parametrize("imported_object", IMPORTS)
def test_induced_polarization(imported_object):
    """
    Test imports from the induced_polarization submodule
    """
    import simpeg.electromagnetics.static.induced_polarization.utils as utils

    assert hasattr(utils, imported_object)
