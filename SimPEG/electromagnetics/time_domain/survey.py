import properties
from ...survey import BaseSurvey
from .sources import BaseTDEMSrc
import numpy as np


####################################################
# Survey
####################################################


class Survey(BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseTDEMSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)

        _source_location_dict = {}
        _source_location_by_sounding_dict = {}
        _source_frequency_by_sounding_dict = {}

        for src in source_list:
            if src.i_sounding not in _source_location_dict:
                _source_location_dict[src.i_sounding] = []
                _source_location_by_sounding_dict[src.i_sounding] = []
            _source_location_dict[src.i_sounding] += [src]
            _source_location_by_sounding_dict[src.i_sounding] += [src.location]

        self._source_location_dict = _source_location_dict
        self._source_location_by_sounding_dict = _source_location_by_sounding_dict

    @property
    def source_location_by_sounding_dict(self):
        """
        Source location in the survey as a dictionary
        """
        return self._source_location_by_sounding_dict

    def get_sources_by_sounding_number(self, i_sounding):
        """
        Returns the sources associated with a specific source location.
        :param float i_sounding: source location number
        :rtype: dictionary
        :return: sources at the sepcified source location
        """
        assert (
            i_sounding in self._source_location_dict
        ), "The requested sounding is not in this survey."
        return self._source_location_dict[i_sounding]

    @property
    def vnD_by_sounding_dict(self):
        if getattr(self, '_vnD_by_sounding_dict', None) is None:
            self._vnD_by_sounding_dict = {}
            for i_sounding in self.source_location_by_sounding_dict:
                source_list = self.get_sources_by_sounding_number(i_sounding)
                nD = 0
                for src in source_list:
                    nD +=src.nD
                self._vnD_by_sounding_dict[i_sounding] = nD
        return self._vnD_by_sounding_dict