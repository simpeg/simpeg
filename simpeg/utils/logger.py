"""
Define global logger for SimPEG.
"""

import logging

__all__ = ["LOGGER"]


def _create_logger():
    """
    Create logger for SimPEG.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("{levelname}: {message}", style="{")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = _create_logger()
