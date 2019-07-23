from scipy.constants import mu_0
import properties

from ...utils import Zero, Identity
from ...survey import BaseSurvey
from .sources import SourceField

import warnings


class MagneticSurvey(BaseSurvey):
    """Base Magnetics Survey"""

    # source_field = properties.Instance(
    #     "The inducing field source for the survey",
    #     properties.Instance("A SimPEG source", SourceField),
    #     default=SourceField
    # )

    def __init__(self, source_field, **kwargs):
        self.source_field = source_field
        BaseSurvey.__init__(self, **kwargs)

    def eval(self, fields):
        return fields

    @property
    def nRx(self):
        return self.source_field.receiver_list[0].locations.shape[0]

    @property
    def receiver_locations(self):
        return self.source_field.receiver_list[0].locations

    @property
    def components(self):
        return self.source_field.receiver_list[0].components

