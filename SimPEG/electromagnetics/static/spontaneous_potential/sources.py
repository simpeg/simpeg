import numpy as np
from SimPEG.electromagnetics.static.resistivity import receivers
from SimPEG import survey
from SimPEG.utils import validate_list_of_types


class StreamingCurrents(survey.BaseSrc):
    # A class that uses dc receivers (pole / dipole)

    def __init__(self, receiver_list, **kwargs):
        location = [np.nan, np.nan, np.nan]
        super().__init__(receiver_list=receiver_list, location=location, **kwargs)

    @property
    def receiver_list(self):
        return self._receiver_list

    @receiver_list.setter
    def receiver_list(self, value):
        self._receiver_list = validate_list_of_types(
            "receiver_list", value, receivers.BaseRx
        )
