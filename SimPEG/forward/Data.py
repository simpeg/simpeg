from SimPEG import utils


class SimPEGData(object):
    """Data holds the observed data, and the standard deviations."""

    __metaclass__ = utils.Save.Savable

    std = None    #: Estimated Standard Deviations
    dobs = None   #: Observed data
    dtrue = None  #: True data, if data is synthetic
    mtrue = None  #: True model, if data is synthetic

    def __init__(self, prob, **kwargs):
        utils.setKwargs(self, **kwargs)
        self.prob = prob

    def isSynthetic(self):
        "Check if the data is synthetic."
        return self.mtrue is not None
