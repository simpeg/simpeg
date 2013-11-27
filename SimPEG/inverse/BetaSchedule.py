

class Cooling(object):
    """Simple Beta Schedule"""

    beta0 = None            #: The initial beta value, set to none means that it will be approximated in the first iteration.
    beta_coolingFactor = 2.

    def getBeta(self):
        if self._beta is None:
            return self.beta0
        return self._beta / self.beta_coolingFactor
