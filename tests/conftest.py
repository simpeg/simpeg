import pytest
from simpeg.utils import get_logger
import logging


@pytest.fixture(scope="session", autouse=True)
def quiet_logger_for_tests():
    logger = get_logger()

    init_level = logger.level
    # default solver log is issued at the INFO level.
    # set the logger to the higher WARNING level to
    # ignore the default solver messages.
    logger.setLevel(logging.WARNING)

    yield

    logger.setLevel(init_level)
