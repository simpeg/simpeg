from ...survey import BaseSurvey
from .sources import BaseTDEMSrc

from ...utils.code_utils import validate_list_of_types

####################################################
# Survey
####################################################


class Survey(BaseSurvey):
    """Time domain electromagnetic survey

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.time_domain.sources.BaseTDEMSrc
        List of SimPEG TDEM sources
    """

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", BaseTDEMSrc),
    #     default=[],
    # )

    def __init__(self, source_list, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)

    @property
    def source_list(self):
        """List of TDEM sources associated with the survey

        Returns
        -------
        list of BaseTDEMSrc
            List of TDEM sources associated with the survey
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list", new_list, BaseTDEMSrc, ensure_unique=True
        )
