import numpy as np
from simpeg.electromagnetics.static.resistivity import receivers
from simpeg import survey
from simpeg.utils import validate_list_of_types


class StreamingCurrents(survey.BaseSrc):
    """A streaming current source.

    Parameters
    ----------
    receiver_list : list of .resistivity.receivers.BaseRx
        The list of Pole and Dipole receivers that listen
        to this source.
    """

    def __init__(self, receiver_list, **kwargs):
        location = [np.nan, np.nan, np.nan]
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)

    @property
    def receiver_list(self):
        """The list of receivers.

        Returns
        -------
        list of .resistivity.receivers.BaseRx
        """
        return self._receiver_list

    @receiver_list.setter
    def receiver_list(self, value):
        self._receiver_list = validate_list_of_types(
            "receiver_list", value, receivers.BaseRx
        )
