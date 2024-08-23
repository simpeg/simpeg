from .doc_inherit import DoceratorMeta


class BaseSimPEG(metaclass=DoceratorMeta):
    """Base class for simpeg classes."""

    # Developer note:
    # This class is mostly used to identify simpeg classes and enable
    # class argument documentation inheritance using the meta class.
