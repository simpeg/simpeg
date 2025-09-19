"""
Define logger for SimPEG.
"""

import logging

__all__ = ["get_logger"]


def _create_logger():
    """
    Create logger for SimPEG.
    """
    logger = logging.getLogger("SimPEG")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("{levelname}: {message}", style="{")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = _create_logger()


def get_logger():
    r"""
    Get the default event logger.

    The logger records events and relevant information while setting up simulations and
    inversions. By default the logger will stream to stderr and using the INFO level.

    Returns
    -------
    logger : :class:`logging.Logger`
        The logger object for SimPEG.
    """
    return LOGGER
