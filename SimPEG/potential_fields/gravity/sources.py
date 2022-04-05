from ...survey import BaseSrc
from ...utils.code_utils import deprecate_class


class SourceField(BaseSrc):
    """Source field for gravity integral formulation

    Parameters
    ----------
    receivers_list : list of SimPEG.potential_fields.receivers.Point
        List of magnetics receivers
    """

    parameters = None

    def __init__(self, receiver_list=None, **kwargs):
        super(SourceField, self).__init__(receiver_list=receiver_list, **kwargs)


@deprecate_class(removal_version="0.16.0", error=True)
class SrcField(SourceField):
    pass
