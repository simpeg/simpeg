from ...survey import BaseSrc
from ...utils.code_utils import deprecate_class


class SourceField(BaseSrc):
    """ Define the inducing field """

    parameters = None

    def __init__(self, receiver_list=None, **kwargs):
        super(SourceField, self).__init__(receiver_list=receiver_list, **kwargs)


@deprecate_class(removal_version="0.15.0")
class SrcField(SourceField):
    pass
