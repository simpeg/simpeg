import properties
import numpy as np
from scipy.constants import mu_0


class SourceField(Survey.BaseSrc):
    """ Define the inducing field """

    param = None  #: Inducing field param (Amp, Incl, Decl)

    def __init__(self, receiver_list=None, **kwargs):
        super(SourceField, self).__init__(receiver_list=receiver_list, **kwargs)
