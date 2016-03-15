from BaseMag import BaseMagSurvey

class MagSurveyIE(BaseMagSurvey):
    """Base Magnetics Survey"""

    rxLoc = None #: receiver locations
    rxType = None #: receiver type

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)

    def setBackgroundField(self, Inc, Dec, Btot):

        Bx = Btot*np.cos(Inc/180.*np.pi)*np.sin(Dec/180.*np.pi)
        By = Btot*np.cos(Inc/180.*np.pi)*np.cos(Dec/180.*np.pi)
        Bz = -Btot*np.sin(Inc/180.*np.pi)

        self.B0 = np.r_[Bx,By,Bz]