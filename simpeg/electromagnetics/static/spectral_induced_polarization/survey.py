import warnings
from ....survey import BaseTimeSurvey
from . import sources
from . import receivers
from .. import resistivity as dc
from ....utils import validate_string


class Survey(BaseTimeSurvey):
    """Spectral IP survey class

    Parameters
    ----------
    source_list : list of simpeg.electromagnetic.static.spectral_induced_polarization.sources.BaseSrc
        List of SimPEG spectral IP sources
    survey_geometry : {"surface", "borehole", "general"}
        Survey geometry.
    """

    _n_pulse = 2
    _T = 8.0

    def __init__(self, source_list=None, survey_geometry="surface", **kwargs):
        if (key := "survey_type") in kwargs:
            warnings.warn(
                f"Argument '{key}' is ignored and will be removed in future "
                "versions of SimPEG. Types of sources and their corresponding "
                "receivers are obtained from their respective classes, without "
                "the need to specify the survey type.",
                FutureWarning,
                stacklevel=1,
            )
            kwargs.pop(key)

        if source_list is None:
            raise AttributeError("Survey cannot be instantiated without sources")
        super(Survey, self).__init__(source_list, **kwargs)
        self.survey_geometry = survey_geometry

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
        """
        ``survey_type`` has been removed.

        Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}

        .. important:

            The `survey_type` property has been removed. Types of sources and
            their corresponding receivers are obtained from their respective
            classes, without the need to specify the survey type.

        Returns
        -------
        str
            Survey type; one of {"dipole-dipole", "pole-dipole", "dipole-pole", "pole-pole"}
        """
        warnings.warn(
            "Property 'survey_type' has been removed."
            "Types of sources and their corresponding receivers are obtained from "
            "their respective classes, without the need to specify the survey type.",
            FutureWarning,
            stacklevel=1,
        )

    @survey_type.setter
    def survey_type(self, var):
        warnings.warn(
            "Property 'survey_type' has been removed."
            "Types of sources and their corresponding receivers are obtained from "
            "their respective classes, without the need to specify the survey type.",
            FutureWarning,
            stacklevel=1,
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
    dc_survey : simpeg.electromagnetics.static.resistivity.survey.Survey
        DC survey object
    times : numpy.ndarray
        Time channels

    Returns
    -------
    simpeg.electromagnetics.static.spectral_induced_polarization.survey.Survey
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
