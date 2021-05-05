import numpy as np
from scipy.constants import mu_0
import properties
from ...utils.code_utils import deprecate_property, deprecate_method

from ...utils import Zero, Identity, uniqueRows
from ..utils import omega
from ...survey import BaseSurvey
from .sources import BaseFDEMSrc


class Survey(BaseSurvey):
    """
    Frequency domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseFDEMSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        # Sort these by frequency
        super(Survey, self).__init__(source_list, **kwargs)

        _frequency_dict = {}
        _source_location_dict = {}
        _source_location_by_sounding_dict = {}
        for src in source_list:
            
            if src.frequency not in _frequency_dict:
                _frequency_dict[src.frequency] = []
            _frequency_dict[src.frequency] += [src]
            
            if src.i_sounding not in _source_location_dict:    
                _source_location_dict[src.i_sounding] = []
                _source_location_by_sounding_dict[src.i_sounding] = []
            _source_location_dict[src.i_sounding] += [src]
            _source_location_by_sounding_dict[src.i_sounding] += [src.location]
            
        self._frequency_dict = _frequency_dict
        self._frequencies = sorted([f for f in self._frequency_dict])
        self._source_location_dict = _source_location_dict
        self._source_location_by_sounding_dict = _source_location_by_sounding_dict
  
    @property
    def frequencies(self):
        """
        Frequencies in the survey
        """
        return self._frequencies

    freqs = deprecate_property(
        frequencies, "freq", new_name="frequencies", removal_version="0.15.0"
    )

    @property
    def num_frequencies(self):
        """Number of frequencies"""
        return len(self._frequency_dict)

    nFreq = deprecate_property(
        num_frequencies, "nFreq", new_name="num_frequencies", removal_version="0.15.0"
    )

    @property
    def num_sources_by_frequency(self):
        """Number of sources at each frequency"""
        if getattr(self, "_num_sources_by_frequency", None) is None:
            self._num_sources_by_frequency = {}
            for freq in self.frequencies:
                self._num_sources_by_frequency[freq] = len(self.getSrcByFreq(freq))
        return self._num_sources_by_frequency

    nSrcByFreq = deprecate_property(
        num_sources_by_frequency,
        "nSrcByFreq",
        new_name="num_sources_by_frequency",
        removal_version="0.15.0",
    )

    def get_sources_by_frequency(self, frequency):
        """
        Returns the sources associated with a specific frequency.
        :param float frequency: frequency for which we look up sources
        :rtype: dictionary
        :return: sources at the sepcified frequency
        """
        assert (
            frequency in self._frequency_dict
        ), "The requested frequency is not in this survey."
        return self._frequency_dict[frequency]

    getSrcByFreq = deprecate_method(get_sources_by_frequency, "getSrcByFreq", "0.15.0")

    @property
    def source_location_by_sounding_dict(self):
        """
        Source locations in the survey as a dictionary
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
    
    @property
    def vnrx_by_sounding_dict(self):
        if getattr(self, '_vnrx_by_sounding_dict', None) is None:
            self._vnrx_by_sounding_dict = {}
            for i_sounding in self.source_location_by_sounding_dict:
                source_list = self.get_sources_by_sounding_number(i_sounding)
                nrx = 0
                for src in source_list:
                    for rx in src.receiver_list:
                        nrx +=len(rx.locations)
                    self._vnrx_by_sounding_dict[i_sounding] = nrx
        return self._vnrx_by_sounding_dict        

    @property
    def frequency_by_sounding_dict(self):
        if getattr(self, "_frequency_by_sounding_dict", None) is None:
            self.get_attributes_by_sounding()
        return self._frequency_by_sounding_dict

    @property
    def receiver_location_by_sounding_dict(self):
        if getattr(self, "_receiver_location_by_sounding_dict", None) is None:
            self.get_attributes_by_sounding()        
        return self._receiver_location_by_sounding_dict

    @property
    def receiver_orientation_by_sounding_dict(self):
        if getattr(self, "_receiver_orientation_by_sounding_dict", None) is None:
            self.get_attributes_by_sounding()        
        return self._receiver_orientation_by_sounding_dict

    @property
    def receiver_use_offset_by_sounding_dict(self):
        if getattr(self, "_receiver_use_offset_by_sounding_dict", None) is None:
            self.get_attributes_by_sounding()        
        return self._receiver_use_offset_by_sounding_dict        

    def get_attributes_by_sounding(self):
        self._frequency_by_sounding_dict = {}
        self._receiver_location_by_sounding_dict = {}
        self._receiver_orientation_by_sounding_dict = {}
        self._receiver_use_offset_by_sounding_dict = {}
        source_location_by_sounding_dict = self.source_location_by_sounding_dict
        for i_sounding in source_location_by_sounding_dict:
            source_list = self.get_sources_by_sounding_number(i_sounding)
            rx_locations = []
            rx_orientations = []
            rx_use_offset = []
            frequencies = []
            for src in source_list:
                for rx in src.receiver_list:
                    rx_locations.append(rx.locations)
                    rx_orientations.append(rx.orientation)
                    rx_use_offset.append(rx.use_source_receiver_offset)
                    frequencies.append(src.frequency)
            self._frequency_by_sounding_dict[i_sounding] = np.hstack([frequencies])
            self._receiver_orientation_by_sounding_dict[i_sounding] = np.hstack([rx_orientations])
            self._receiver_location_by_sounding_dict[i_sounding] = np.vstack([rx_locations])[:,0,:]
            self._receiver_use_offset_by_sounding_dict[i_sounding] = np.hstack([rx_use_offset])