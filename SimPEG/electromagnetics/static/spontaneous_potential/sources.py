from ....survey import BaseSrc

class SpontaneousPotentialSource(BaseSrc):

    def __init__(self, receivers_list, **kwargs):
        BaseSrc.__init__(self, receivers_list, **kwargs)
        
