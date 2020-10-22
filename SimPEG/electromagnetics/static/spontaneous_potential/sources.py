from ....survey import BaseSrc

class SpontaneousPotentialSource(BaseSrc):
    """
    Source class for spontaneous potential simulations

    For spontaneous potential problems, the source represents the
    model parameters for the simulation. However to fit with the
    SimPEG structure of defining surveys, we have chosen to define
    an empty source class. This class may be given special properties
    or methods in the future.
    """

    def __init__(self, receivers_list, **kwargs):
        BaseSrc.__init__(self, receivers_list, **kwargs)
        
