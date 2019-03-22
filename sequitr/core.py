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
__email__ = 'code@arlowe.co.uk'
__version__ = '0.1.7'

# import the config parser
import os
import ConfigParser
from ast import literal_eval



DEFAULT_LOGGER_PROCESSES = ('server_process', 'worker_process')



class ServerConfiguration:
    """ A server config """
    VERSION = __version__
    LOGDIR = ''
    JOBDIR = ''
    OUTDIR = ''
    SERVER_IP = ''
    DEFAULT_GPUS = [0,1,2,3]
    MAX_PROCESSES = 4
    DELAY = 60
    LOCAL = True
    VERBOSE_LOG = True
    CORES = 0

class TensorflowConfiguration:
    """ Tensorflow config """
    TF_LOG_LEVEL = '3'
    LOGDIR = ''
    MODELDIR = ''
    LOG_DEVICE_PLACEMENT = True
    ALLOW_GROWTH = True


def _configure(config_file='server.config'):
    """ Configure the package from a config file """

    if not os.path.exists(config_file):
        logger.error('Cannot find server config file {0:s}'
                     .format(config_file))
        return

    config = ConfigParser.ConfigParser()
    config.read(config_file)

    # set server config
    for opt in config.options('config'):
        val = _get_config_opt_correct_type(config, 'config', opt)
        setattr(ServerConfiguration, opt.upper(), val)

    # set tensorflow config
    for opt in config.options('tensorflow'):
        val = _get_config_opt_correct_type(config, 'tensorflow', opt)
        setattr(TensorflowConfiguration, opt.upper(), val)

    # finally set the list of devices
    ServerConfiguration.CPUS = []
    ServerConfiguration.GPUS = []

    for cpu in config.options('cpu'):
        ServerConfiguration.CPUS += [config.get('cpu', cpu)]
    for gpu in config.options('gpu'):
        ServerConfiguration.GPUS += [config.get('gpu', gpu)]

    return ServerConfiguration.VERSION




def _get_config_opt_correct_type(config, section, opt):
    """ Return correctly typed configuration info """

    BOOL_OPTS = ('LOCAL', 'VERBOSE_LOG', 'LOG_DEVICE_PLACEMENT', 'ALLOW_GROWTH')
    INT_OPTS = ('MAX_PROCESSES', 'DELAY', 'CORES')
    LIST_OPTS = ('DEFAULT_GPUS')

    if opt.upper() in BOOL_OPTS:
        return config.getboolean(section, opt)
    elif opt.upper() in INT_OPTS:
        return config.getint(section, opt)
    elif opt.upper() in LIST_OPTS:
        return literal_eval(config.get(section, opt))
    else:
        return config.get(section, opt)




# call the configuration
_configure()

if __name__ == '__main__':
    pass
