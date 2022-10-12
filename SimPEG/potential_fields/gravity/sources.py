from ...survey import BaseSrc


class SourceField(BaseSrc):
    """Source field for gravity integral formulation

    Parameters
    ----------
    receivers_list : list of SimPEG.potential_fields.receivers.Point
        List of magnetics receivers
    """

    def __init__(self, receiver_list=None, **kwargs):
        super(SourceField, self).__init__(receiver_list=receiver_list, **kwargs)
