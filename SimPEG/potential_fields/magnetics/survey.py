from scipy.constants import mu_0
import properties

from ...utils import Zero, Identity
from ...survey import BaseSurvey
from .sources import SourceField

import warnings


class MagneticSurvey(BaseSurvey):
    """Base Magnetics Survey"""

    rx_type = None  #: receiver type

    def __init__(self, sourceField, **kwargs):
        self.sourceField = sourceField
        BaseSurvey.__init__(self, **kwargs)

    def eval(self, fields):
        return fields

    @property
    def nRx(self):
        return self.sourceField.receiver_locations.shape[0]

    @property
    def receiver_locations(self):
        return self.sourceField.receiver_locations
