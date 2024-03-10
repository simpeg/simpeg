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

    def __init__(self, source_list, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)

        _source_location = {}
        _source_location_by_sounding = {}

        for src in source_list:
            if src.i_sounding not in _source_location:
                _source_location[src.i_sounding] = []
                _source_location_by_sounding[src.i_sounding] = []
            _source_location[src.i_sounding] += [src]
            _source_location_by_sounding[src.i_sounding] += [src.location]

        self._source_location = _source_location
        self._source_location_by_sounding = _source_location_by_sounding

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

    @property
    def source_location_by_sounding(self):
        """
        Source location in the survey as a dictionary
        """
        return self._source_location_by_sounding

    def get_sources_by_sounding_number(self, i_sounding):
        """
        Returns the sources associated with a specific source location.
        :param float i_sounding: source location number
        :rtype: dictionary
        :return: sources at the sepcified source location
        """
        assert (
            i_sounding in self._source_location
        ), "The requested sounding is not in this survey."
        return self._source_location[i_sounding]

    @property
    def vnD_by_sounding(self):
        if getattr(self, "_vnD_by_sounding", None) is None:
            self._vnD_by_sounding = {}
            for i_sounding in self.source_location_by_sounding:
                source_list = self.get_sources_by_sounding_number(i_sounding)
                nD = 0
                for src in source_list:
                    nD += src.nD
                self._vnD_by_sounding[i_sounding] = nD
        return self._vnD_by_sounding
