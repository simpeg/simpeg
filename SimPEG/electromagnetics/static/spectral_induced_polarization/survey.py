import numpy as np

# import properties

from ....survey import BaseTimeSurvey
from . import sources
from . import receivers
from .. import resistivity as dc
from ....utils import validate_string


class Survey(BaseTimeSurvey):
    """Spectral IP survey class

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.static.spectral_induced_polarization.sources.BaseSrc
        List of SimPEG spectral IP sources
    survey_geometry : {"surface", "borehole", "general"}
        Survey geometry.
    survey_type : {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        Survey type.
    """

    _n_pulse = 2
    _T = 8.0

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", sources.BaseSrc),
    #     default=[],
    # )

    # def __init__(self, source_list=None, **kwargs):
    #     super().__init__(source_list, **kwargs)
    def __init__(
        self,
        source_list=None,
        survey_geometry="surface",
        survey_type="dipole-dipole",
        **kwargs
    ):
        if source_list is None:
            raise AttributeError("Survey cannot be instantiated without sources")
        super(Survey, self).__init__(source_list, **kwargs)
        self.survey_geometry = survey_geometry
        self.survey_type = survey_type

    @property
    def n_pulse(self):
        """Number of pulses

        Returns
        -------
        int
            Number of pulses
        """
        return self._n_pulse

    @property
    def T(self):
        """Period

        Returns
        -------
        float
            Period
        """
        return self._T

    @property
    def survey_geometry(self):
        """Survey geometry; one of {"surface", "borehole", "general"}

        Returns
        -------
        str
            Survey geometry; one of {"surface", "borehole", "general"}
        """
        return self._survey_geometry

    @survey_geometry.setter
    def survey_geometry(self, var):
        self._survey_geometry = validate_string(
            "survey_geometry", var, ("surface", "borehole", "general")
        )

    @property
    def survey_type(self):
        """Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}

        Returns
        -------
        str
            Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        """
        return self._survey_type

    @survey_type.setter
    def survey_type(self, var):
        self._survey_type = validate_string(
            "survey_type",
            var,
            ("dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"),
        )

    @property
    def n_locations(self):
        """Number of data locations

        Returns
        -------
        int
            Number of data locations
        """
        return int(self.nD // self.unique_times.size)


def from_dc_to_sip_survey(survey_dc, times):
    """Create SIP survey from DC survey geometry

    Parameters
    ----------
    dc_survey : SimPEG.electromagnetics.static.resistivity.survey.Survey
        DC survey object
    times : numpy.ndarray
        Time channels

    Returns
    -------
    SimPEG.electromagnetics.static.spectral_induced_polarization.survey.Survey
        An SIP survey object
    """
    source_list = survey_dc.source_list

    source_list_sip = []
    for src in source_list:
        receiver_list_sip = []
        for rx in src.receiver_list:
            if isinstance(rx, dc.receivers.Pole):
                rx_sip = receivers.Pole(rx.locations, times=times)
            elif isinstance(rx, dc.receivers.Dipole):
                rx_sip = receivers.Dipole(rx.locations[0], rx.locations[1], times=times)
            else:
                print(rx)
                raise NotImplementedError()
            receiver_list_sip.append(rx_sip)

        if isinstance(src, dc.sources.Pole):
            src_sip = sources.Pole(receiver_list_sip, src.loc)
        elif isinstance(src, dc.sources.Dipole):
            src_sip = sources.Dipole(receiver_list_sip, src.loc[0], src.loc[1])
        else:
            print(src)
            raise NotImplementedError()
        source_list_sip.append(src_sip)

    survey_sip = Survey(source_list_sip)

    return survey_sip
