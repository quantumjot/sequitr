#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:     Sequitr
# Purpose:  Sequitr is a small, lightweight Python library for common image
#           processing tasks in optical microscopy, in particular, single-
#           molecule imaging, super-resolution or time-lapse imaging of cells.
#           Sequitr implements fully convolutional neural networks for image
#           segmentation and classification. Modelling of the PSF is also
#           supported, and the library is designed to integrate with
#           BayesianTracker.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  23/03/2018
#-------------------------------------------------------------------------------

__author__ = 'Alan R. Lowe'
__email__ = 'a.lowe@ucl.ac.uk'

import os
import core
import logging
import functools

from datetime import datetime
from time import gmtime, strftime

def generate_log_filename():
    """ Return a timestamped log filename """
    return "LOG_"+strftime("(%Y-%m-%d)_%H-%M-%S", gmtime())+".txt"



def setup_logging(filepath=core.ServerConfiguration.LOGDIR,
                  log_name='server_process'):
    """ setup_logging

    Set up logging to file for the server process.

    Args:
        filepath: the path to the output directory
        log_name: the name of the log process

    Returns:
        log_file: the filename of the log file
    """

    if not os.path.exists(filepath):
        raise IOError('LOG_DIR filepath does not exist: {0:s}'.format(filepath))

    if not log_name in core.DEFAULT_LOGGER_PROCESSES:
        raise ValueError('Log_name should be in {0:s}.'
                         .format(core.DEFAULT_LOGGER_PROCESSES))

    filename = generate_log_filename()
    log_file = os.path.join(filepath, filename)

    # configure log formatter
    log_fmt = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s',
                                datefmt='%Y/%m/%d %I:%M:%S %p')

    # configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_fmt)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_fmt)

    # setup a server log, add file and stream handlers
    logger = logging.getLogger(log_name)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    return log_file


def get_logger(loggers=core.DEFAULT_LOGGER_PROCESSES):
    """ Get the appropriate logger, whether it's the server process
    or the worker process.
    """

    for l in loggers:
        logger = logging.getLogger(l)
        if logger.handlers:
            return logger

    return None











def exception_logger(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur.

    Notes:
        https://www.blog.pythonlibrary.org/2016/06/09/
        python-how-to-create-an-exception-logging-decorator/

    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        try:
            # try to call the function
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in: {0:s}".format(function.__name__)

            if logger is not None:
                logger.exception(err)
            else:
                print err
    return wrapper





if __name__ == '__main__':
    pass
