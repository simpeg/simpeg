from ....survey import BaseSurvey

class Survey(BaseSurvey):

    def __init__(self, source_list, **kwargs):
        BaseSurvey.__init__(self, source_list, **kwargs)
        
