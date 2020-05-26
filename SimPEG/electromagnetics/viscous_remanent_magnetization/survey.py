import numpy as np
import properties

from ...survey import BaseSurvey
from .sources import BaseSrcVRM


############################################
# BASE VRM SURVEY CLASS
############################################


class SurveyVRM(BaseSurvey):

    """

    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseSrcVRM),
        default=[],
    )

    t_active = properties.Array(
        "Boolean array where True denotes active data in the inversion", dtype=bool
    )

    def __init__(self, source_list=None, **kwargs):

        t_active = kwargs.pop("t_active", None)

        super(SurveyVRM, self).__init__(source_list=source_list, **kwargs)

        self._nD_all = self.vnD.sum()
        self._nD = self._nD_all

        if t_active is None:
            self.t_active = np.ones(self._nD_all, dtype=bool)
        else:
            self.t_active = t_active

    @properties.validator("t_active")
    def _t_active_validator(self, change):
        if self._nD_all != len(change["value"]):
            raise ValueError(
                "Length of t_active boolean array must equal number of data. Number of data is %i"
                % self._nD_all
            )

    @property
    def nD(self):
        if self._nD is None:
            self._nD = np.sum(self.t_active)
        return self._nD

    def set_active_interval(self, tmin, tmax):
        """Set active times using an interval"""

        srcList = self.source_list
        nSrc = len(srcList)
        tActBool = np.array([])

        for pp in range(0, nSrc):

            rxList = srcList[pp].receiver_list
            nRx = len(rxList)

            for qq in range(0, nRx):

                times = rxList[qq].times
                nLoc = np.shape(rxList[qq].locations)[0]
                tActBool = np.r_[tActBool, np.kron(np.ones(nLoc), times)]

        self.t_active = (tActBool >= tmin) & (tActBool <= tmax)
        self._nD = np.sum(self.t_active)
