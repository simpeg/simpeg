from ....survey import BaseSurvey

class Survey(BaseSurvey):
    """
    Survey class for spontaneous potential simulations
    """ 

    def __init__(self, source_list, **kwargs):
        BaseSurvey.__init__(self, source_list, **kwargs)
        