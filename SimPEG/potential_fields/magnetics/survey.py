from scipy.constants import mu_0
import properties

from ...utils import Zero, Identity
from ..utils import omega
from ...survey import BaseSurvey
from .sources import BaseFDEMSrc

import warnings


class Survey(BaseSurvey):
    """Base Magnetics Survey"""

    rx_type = None  #: receiver type

    def __init__(self, sourceField, **kwargs):
        self.sourceField = sourceField
        Survey.BaseSurvey.__init__(self, **kwargs)

    def eval(self, fields):
        return fields

    @property
    def nD(self):

        nD = 0
        for receiver in self.sourceField.receiver_list:
            nD += receiver.nD

        return nD

    @property
    def nRx(self):
        return self.sourceField.receiver_locations.shape[0]

    @property
    def receiver_locations(self):
        return self.sourceField.receiver_locations

    @property
    def receiver_index(self):
        return self.sourceField.receiver_index
