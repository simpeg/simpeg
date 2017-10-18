from SimPEG import Survey
from SimPEG.VRM import RxVRM, SrcVRM


############################################
# BASE VRM SURVEY CLASS
############################################

class SurveyVRM(Survey.BaseSurvey):

    def __init__(self, srcList, **kwargs):
        super(SurveyVRM, self).__init__(**kwargs)
        self.srcList = srcList
        self.srcPair = SrcVRM.BaseSrcVRM
        self.rxPair = RxVRM.BaseRxVRM

    def dpred(self, m):

        """Predict data for a given model."""

        assert self.ispaired, "Survey must be paired with a VRM problem"

        return self.prob.fields(m, fType=None)
