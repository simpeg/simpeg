import pytest
from simpeg.utils import get_logger
import logging


@pytest.fixture(scope="session", autouse=True)
def quiet_logger_for_tests(request):
    logger = get_logger()

    init_level = logger.level
    # default solver log is issued at the INFO level.
    # set the logger to the higher WARNING level to
    # ignore the default solver messages.
    logger.setLevel(logging.WARNING)

    yield

    logger.setLevel(init_level)


@pytest.fixture()
def info_logging():
    # provide a fixture to temporarily set the logging level to info
    logger = get_logger()
    init_level = logger.level
    logger.setLevel(logging.INFO)

    yield

    logger.setLevel(init_level)
