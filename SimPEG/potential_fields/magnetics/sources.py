from ...survey import BaseSrc
from .analytics import IDTtoxyz


class SourceField(BaseSrc):
    """Source field for magnetics integral formulation

    Parameters
    ----------
    receivers_list : list of SimPEG.potential_fields.receivers.Point
        List of magnetics receivers
    parameters : (3) array_like of float
        Define the Earth's inducing field according to
        [*amplitude*, *inclination*, *declination*] where:

        - *amplitude* is the field intensity in nT
        - *inclination* is the inclination of the Earth's field in degrees
        - *declination* is the declination of the Earth's field in degrees
    """

    def __init__(self, receiver_list=None, parameters=[50000, 90, 0], **kwargs):
        assert (
            len(parameters) == 3
        ), "Inducing field 'parameters' must be a list or tuple of length 3 (amplitude, inclination, declination"

        self.parameters = parameters
        self.b0 = IDTtoxyz(-parameters[1], parameters[2], parameters[0])
        super(SourceField, self).__init__(
            receiver_list=receiver_list, parameters=parameters, **kwargs
        )
