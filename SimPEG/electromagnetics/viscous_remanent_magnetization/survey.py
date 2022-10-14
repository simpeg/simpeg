import numpy as np

# import properties

from ...survey import BaseSurvey
from .sources import BaseSrcVRM
from ...utils import validate_list_of_types, validate_ndarray_with_shape


############################################
# BASE VRM SURVEY CLASS
############################################


class SurveyVRM(BaseSurvey):
    """Base VRM survey

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.viscous_remanent_magnetization.sources.BaseSrcVRM
        List of SimPEG VRM sources
    t_active : numpy.ndarray of bool
        Active time channels used in inversion
    """

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", BaseSrcVRM),
    #     default=[],
    # )

    # t_active = properties.Array(
    #     "Boolean array where True denotes active data in the inversion", dtype=bool
    # )

    def __init__(self, source_list, t_active=None, **kwargs):

        super(SurveyVRM, self).__init__(source_list=source_list, **kwargs)

        self._nD_all = self.vnD.sum()
        self._nD = self._nD_all

        if t_active is None:
            t_active = np.ones(self._nD_all, dtype=bool)
        self.t_active = validate_ndarray_with_shape(
            "t_active", t_active, shape=(self._nD_all,), dtype=bool
        )

    @property
    def source_list(self):
        """List of VRM sources associated with the survey

        Returns
        -------
        list of BaseSrcVRM
            List of VRM sources associated with the survey
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        self._source_list = validate_list_of_types(
            "source_list", new_list, BaseSrcVRM, ensure_unique=True
        )

    # @properties.validator("t_active")
    # def _t_active_validator(self, change):
    #     if self._nD_all != len(change["value"]):
    #         raise ValueError(
    #             "Length of t_active boolean array must equal number of data. Number of data is %i"
    #             % self._nD_all
    #         )

    @property
    def nD(self):
        if self._nD is None:
            self._nD = np.sum(self.t_active)
        return self._nD

    def set_active_interval(self, tmin, tmax):
        """Set active times using an interval"""

        source_list = self.source_list
        nSrc = len(source_list)
        tActBool = np.array([])

        for pp in range(0, nSrc):

            receiver_list = source_list[pp].receiver_list
            nRx = len(receiver_list)

            for qq in range(0, nRx):

                times = receiver_list[qq].times
                nLoc = np.shape(receiver_list[qq].locations)[0]
                tActBool = np.r_[tActBool, np.kron(np.ones(nLoc), times)]

        self.t_active = (tActBool >= tmin) & (tActBool <= tmax)
        self._nD = np.sum(self.t_active)
