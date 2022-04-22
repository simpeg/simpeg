import numpy as np
# import properties

from ....survey import BaseTimeSurvey
from . import sources
from . import receivers
from .. import resistivity as dc


class Survey(BaseTimeSurvey):
    """Spectral IP survey class

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.static.spectral_induced_polarization.sources.BaseSrc
        List of SimPEG spectral IP sources
    survey_geometry : str, default="surface"
        Survey geometry. Choose one of {"surface", "borehole", "general"}
    survey_type : str, default="dipole-dipole"
        Survey type. Choose one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
    """

    n_pulse = 2
    T = 8.0

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", sources.BaseSrc),
    #     default=[],
    # )

    # def __init__(self, source_list=None, **kwargs):
    #     super().__init__(source_list, **kwargs)
    def __init__(self, source_list=None, survey_geometry="surface", survey_type="dipole-dipole", **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)
        self.survey_geometry = survey_geometry
        self.survey_type = survey_type

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

        if isinstance(var, str):
            if var.lower() in ("surface", "borehole", "general"):
                self._survey_geometry = var.lower() 
            else:
                raise ValueError(f"'survey_geometry' must be 'surface', 'borehole' or 'general'. Got {var}")
        else:
            raise TypeError(f"'survey_geometry' must be a str. Got {type(var)}")

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

        if isinstance(var, str):
            if var.lower() in ("dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"):
                self._survey_type = var.lower() 
            else:
                raise ValueError(f"'survey_type' must be 'dipole-dipole', 'pole-dipole', 'dipole-pole', 'pole-pole'. Got {var}")
        else:
            raise TypeError(f"'survey_type' must be a str. Got {type(var)}")


    @property
    def n_locations(self):
        """Number of data locations

        Returns
        -------
        int
            Number of data locations
        """
        return int(self.nD / self.unique_times.size)


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
    SimPEG.electromagnetics.static.induced_polarization.survey.Survey
        An IP survey object
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
