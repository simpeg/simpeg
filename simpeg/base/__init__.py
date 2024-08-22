from .doc_inherit import DoceratorMeta


class BaseSimPEG(metaclass=DoceratorMeta):
    """Base class for simpeg classes."""

    # Developer note:
    # This class is mostly used to identify simpeg classes
    # and to catch any leftover keyword arguments before calling
    # object.__init__() (if that was the next class on the mro above
    # this one). If there are any leftover arguments, it throws a TypeError
    # with an appropriate message reference the class that was initialized.
