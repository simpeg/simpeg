from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging

import properties

# Define a custom logger class
class simpeg_logger(properties.HasProperties):
    """Extension of the logging module to work with SimPEG"""

    # Constant properties are only set in the __init__ call
    @property
    def logger_name(self):
        return self._logger.name

    @property
    def file_name(self):
        return self._file_handler.get_name()


    def _ensure_path_exists(self, path_str):
        """Make sure the path exists

        :param string path_str: Path to the file
        """
        path_name = os.path.dirname(path_str)
        if path_name is "":
            return True
        return os.path.exists(path_name)


    file_level = properties.StringChoice(
        'Level of logging detail to be written to the file',
        choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'],
        required=True
    )
    @properties.observer('file_level')
    def _set_file_handler_level(self, change):
        self._file_handler.setLevel(self._level_dict[change['value']])

    stream_level = properties.StringChoice(
        'Level of logging detail to be written to std_out',
        choices=['DEBUG', 'INFO', 'WARNING','ERROR','CRITICAL'],
        required=True
    )
    @properties.observer('stream_level')
    def _set_stream_handler_level(self, change):
        self._stream_handler.setLevel(self._level_dict[change['value']])

    _level_dict = {
        'DEBUG':logging.DEBUG,
        'INFO':logging.INFO,
        'WARNING':logging.WARNING,
        'ERROR':logging.ERROR,
        'CRITICAL':logging.CRITICAL
    }

    def __init__(self, logger_name, file_name='log_file.log'):
        """
        Initiate a simpeg_logger

        :param string logger_name: Unique name of the logger
        :param string file_name ('log_file.log'): Path to the file to log to
        """
        super(simpeg_logger, self).__init__()
        # Setup input values as getable properties

        # Make the logger
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        if len(self._logger.handlers) > 0:
            self._logger.handlers = []
        # Create a file handler
        if self._ensure_path_exists(file_name):
            print(file_name)
            self._file_handler = logging.FileHandler(file_name)
            self._file_handler.set_name(file_name)
        else:
            raise IOError('Filepath {} does not exist'.format(file_name))
        self.file_level = 'DEBUG'
        # Create a stream handler
        self._stream_handler = logging.StreamHandler()
        self.stream_level = 'INFO'
        # Create a formatter and set to the handlers
        self._formatter = logging.Formatter(
            '%(asctime)s - %(name)s - ' +
            '%(levelname)s - %(message)s'
        )
        self._file_handler.setFormatter(self._formatter)
        self._stream_handler.setFormatter(self._formatter)
        # Set the handlers to the logger
        self._logger.addHandler(self._file_handler)
        self._logger.addHandler(self._stream_handler)


    # Define logging methods
    def debug(self, logging_message):
        """
        Method for logging debugging messages

        :param string logging_message: Descriptive message
        """
        self._logger.debug(logging_message)

    def info(self, logging_message):
        """
        Method for logging information messages

        :param string logging_message: Descriptive message
        """
        self._logger.info(logging_message)

    def warning(self, logging_message):
        """
        Method for logging warning messages

        :param string logging_message: Descriptive message
        """
        self._logger.warning(logging_message)

    def error(self, logging_message):
        """
        Method for logging error messages

        :param string logging_message: Descriptive message
        """
        self._logger.error(logging_message)

    def critical(self, logging_message):
        """
        Method for logging critical messages

        :param string logging_message: Descriptive message
        """
        self._logger.critical(logging_message)
