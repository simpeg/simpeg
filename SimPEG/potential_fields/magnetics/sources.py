from ...survey import BaseSrc
from .analytics import IDTtoxyz


class SourceField(BaseSrc):
    """Define the inducing field"""

    def __init__(
        self,
        receiver_list=None,
        parameters=None,
        amplitude=50000,
        inclination=90,
        declination=0,
        **kwargs
    ):
        if parameters is not None:
            if len(parameters) != 3:
                raise ValueError(
                    "Inducing field 'parameters' must be a list or tuple of length 3 (amplitude, inclination, declination"
                )
            amplitude, inclination, declination = parameters

        self.amplitude = amplitude
        self.inclination = inclination
        self.declination = declination

        self.b0 = IDTtoxyz(-parameters[1], parameters[2], parameters[0])
        super(SourceField, self).__init__(
            receiver_list=receiver_list, parameters=parameters, **kwargs
        )
