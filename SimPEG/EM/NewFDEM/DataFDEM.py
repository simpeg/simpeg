import properties

from ...Data import Data as BaseData
from .SurveyFDEM import Survey


class Data(BaseData):
    """
    A frequency domain data object
    """

    survey = properties.Instance(
        "a frequency domain EM survey",
        Survey
    )

    def __init__(self, **kwargs):
        super(Data, self).__init__(**kwargs)

    @property
    def freqs(self):
        return self.survey.freqs

    @property
    def nFreqs(self):
        return self.survey.nFreqs
