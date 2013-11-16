

class Cooling(object):
    """Simple Beta Schedule"""

    beta0 = 1.e6
    beta_coolingFactor = 5.

    def getBeta(self):
        if self._beta is None:
            return self.beta0
        return self._beta / self.beta_coolingFactor
