from ..resistivity import Survey
from ..resistivity import receivers, sources


def from_dc_to_ip_survey(dc_survey, dim="2.5D"):
    srcList = dc_survey.source_list
    ip_survey = Survey(srcList)

    return ip_survey
