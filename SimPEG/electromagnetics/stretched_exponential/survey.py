from SimPEG import survey

class SESurvey(survey.BaseSurvey):

    def __init__(self, **kwargs):
        survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def nD(self):
        self._nD = self.dobs.size
        return self._nD
