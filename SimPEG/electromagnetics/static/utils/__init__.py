from .static_utils import (
    electrode_separations,
    source_receiver_midpoints,
    pseudo_locations,
    geometric_factor,
    apparent_resistivity_from_voltage,
    apparent_resistivity,
    plot_2d_pseudosection,
    generate_dcip_survey,
    gen_DCIPsurvey,
    # generate_dcip_survey_line,
    generate_dcip_sources_line,
    generate_survey_from_abmn_locations,
    writeUBC_DCobs,
    writeUBC_DClocs,
    convert_survey_3d_to_2d_lines,
    # convertObs_DC3D_to_2D,
    readUBC_DC2Dpre,
    readUBC_DC3Dobs,
    xy_2_lineID,
    r_unit,
    getSrc_locs,
    gettopoCC,
    drapeTopotoLoc,
    genTopography,
    closestPointsGrid,
    gen_3d_survey_from_2d_lines,
    plot_1d_layer_model,
    plot_layer
)

# Import if user has plotly
try:
    from .static_utils import plot_3d_pseudosection
except:
    pass
