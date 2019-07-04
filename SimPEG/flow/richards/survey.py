import numpy as np
import scipy.sparse as sp
import properties
import warnings

from ...survey import BaseSurvey, BaseRx
from ... import utils


class Survey(BaseSurvey):
    """RichardsSurvey"""

    receiver_list = properties.List(
        "list of receivers for flow simulations",
        properties.Instance("A receiver instance", BaseRx)
    )

    def __init__(self, receiver_list, **kwargs):
        self.receiver_list = receiver_list
        BaseSurvey.__init__(self, **kwargs)

    @property
    def nD(self):
        return np.array([rx.nD for rx in self.receiver_list]).sum()

    def deriv(self, simulation, f, du_dm_v=None, v=None):
        """The Derivative with respect to the model."""
        dd_dm = [
            rx.deriv(f, simulation, du_dm_v=du_dm_v, v=v, adjoint=False)
            for rx in self.receiver_list
        ]
        return np.concatenate(dd_dm)

    def derivAdjoint(self, simulation, f, v=None):
        """The adjoint derivative with respect to the model."""
        dd_du = list(range(len(self.receiver_list)))
        dd_dm = list(range(len(self.receiver_list)))
        cnt = 0
        for ii, rx in enumerate(self.receiver_list):
            dd_du[ii], dd_dm[ii] = rx.deriv(
                f, simulation, v=v[cnt:cnt + rx.nD], adjoint=True
            )
            cnt += rx.nD
        return np.sum(dd_du, axis=0), np.sum(dd_dm, axis=0)

    @property
    def rxList(self):
        warnings.warn(
            "source.rxList will be deprecaited and replaced with "
            "source.receiver_list. Please update your code accordingly"
        )
        return self.receiver_list

    @rxList.setter
    def rxList(self, value):
        warnings.warn(
            "source.rxList will be deprecaited and replaced with "
            "source.receiver_list. Please update your code accordingly"
        )
        self.receiver_list = value
