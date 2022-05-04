"""
====================================================================================================
Static Utilities (:mod:`SimPEG.electromagnetics.utils`)
====================================================================================================
.. currentmodule:: SimPEG.electromagnetics.static.utils


.. autosummary::
  :toctree: generated/

  static_utils.electrode_separations
  static_utils.source_receiver_midpoints
  static_utils.pseudo_locations
  static_utils.geometric_factor
  static_utils.apparent_resistivity_from_voltage
  static_utils.apparent_resistivity
  static_utils.plot_pseudosection
  static_utils.generate_dcip_survey
  static_utils.generate_dcip_survey_line
  static_utils.gen_DCIPsurvey
  static_utils.generate_dcip_sources_line
  static_utils.generate_survey_from_abmn_locations
  static_utils.writeUBC_DCobs
  static_utils.writeUBC_DClocs
  static_utils.convert_survey_3d_to_2d_lines
  static_utils.convertObs_DC3D_to_2D
  static_utils.readUBC_DC2Dpre
  static_utils.readUBC_DC3Dobs
  static_utils.xy_2_lineID
  static_utils.r_unit
  static_utils.getSrc_locs
  static_utils.gettopoCC
  static_utils.drapeTopotoLoc
  static_utils.genTopography
  static_utils.closestPointsGrid
  static_utils.gen_3d_survey_from_2d_lines
  static_utils.plot_1d_layer_model
  static_utils.plot_layer

"""
from .static_utils import (
    electrode_separations,
    source_receiver_midpoints,
    pseudo_locations,
    geometric_factor,
    apparent_resistivity_from_voltage,
    apparent_resistivity,
    plot_pseudosection,
    generate_dcip_survey,
    generate_dcip_survey_line,
    gen_DCIPsurvey,
    generate_dcip_sources_line,
    generate_survey_from_abmn_locations,
    writeUBC_DCobs,
    writeUBC_DClocs,
    convert_survey_3d_to_2d_lines,
    convertObs_DC3D_to_2D,
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
    plot_layer,
    plot_pseudoSection,
)

# Import if user has plotly
try:
    from .static_utils import plot_3d_pseudosection
except:
    pass
