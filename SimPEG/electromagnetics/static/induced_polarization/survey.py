from ..resistivity import Survey
from ..resistivity import receivers, sources


def from_dc_to_ip_survey(dc_survey, dim="2.5D"):
    source_list = dc_survey.source_list
    ip_survey = Survey(source_list)

    return ip_survey
