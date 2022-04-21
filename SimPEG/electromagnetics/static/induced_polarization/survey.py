from ..resistivity import Survey
from ..resistivity import receivers, sources


def from_dc_to_ip_survey(dc_survey, dim="2.5D"):
    """Create IP survey from DC survey geometry

    Parameters
    ----------
    dc_survey : SimPEG.electromagnetics.static.resistivity.survey.Survey
        DC survey object
    dim : str, default = "2.5D"
        Dimension of the surface. Choose from {'1D', '2.5D', '3D'}

    Returns
    -------
    SimPEG.electromagnetics.static.induced_polarization.survey.Survey
        An IP survey object
    """
    source_list = dc_survey.source_list
    ip_survey = Survey(source_list)

    return ip_survey
