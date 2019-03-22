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

import sys
import os
import argparse
import ConfigParser
import importlib
from ast import literal_eval
import time

import core
import serverlogs
from utils import check_and_makedir


# # get the server config
# ServerConfiguration = core.ServerConfiguration





@serverlogs.exception_logger
def parse_job_file(filename, header_only=False):
    """ parse_job_file

    Parse a job file. This does a preliminary parse to make
    dure it's a valid job file.  Other parameters are checked at run
    time. Jobs have the following basic structure:

        [job]
        complete = False
        id = 467e3c034f84acbf3d5d955e93358043
        user = Alan
        priority = 99
        time = (2018-09-28)_10-59-02
        module = jobs
        func = SERVER_segment_and_classify
        device = GPU
        params = {'test': 'x'}
        options = {'option': True}


    Args:
        filename: full path to job file
        header_only: read only the header of the job description, excludes the
            parameters and options arguments which fully specify the job. This
            enables better error logging in the worker process.

    Returns:
        job: A JobWrapper object, instantiated with the job description
    """

    if not isinstance(filename, basestring):
        raise Exception("Job filename is not correctly formed")

    if not filename.endswith('.job'):
        raise IOError('Job {0:s} does not have .job file extenstion'
                      .format(filename))

    job_config = ConfigParser.ConfigParser()

    # read the job config
    job_config.read(filename)

    # set up the new job here
    job = JobWrapper(ID=job_config.get('job','ID'),
                     filename=filename,
                     owner=job_config.get('job','user'),
                     priority=job_config.get('job','priority'),
                     device=job_config.get('job','device'))

    # get the details from the job config and convert to the correct types
    job._module = job_config.get('job','module')
    job._func = job_config.get('job','func')

    # if we are only returning the header:
    if header_only: return job

    # check the params and options
    job._params = literal_eval(job_config.get('job','params'))
    if job_config.has_option('job','options'):
        job._options = literal_eval(job_config.get('job','options'))

    # if we have a path to an additional library, check it and append
    if job_config.has_option('job','lib_path'):
        lib_path = job_config.get('job','lib_path')
        if os.path.exists(lib_path): job._lib_path = lib_path

    return job





class JobWrapper(object):
    """ JobWrapper

    Wrapper for jobs.  Takes care of properly specifying jobs, exceuting them,
    logging and completion.

    Comprehensive parsing of the inputs is taken care of elsewhere.

    Args:
        ID: the job ID number
        filename: the job filename, including path
        owner: the owner of the job, e.g. 'Alan'
        priority: the job priority
        device: the device to run the job on ['CPU', 'GPU']

    Properties:

    Members:

    Notes:
        None

    """
    def __init__(self,
                 ID=None,
                 filename=None,
                 owner='root',
                 priority=99,
                 device='CPU'):


        if device not in ['CPU','GPU']:
            logging.warning('Device {0:s} not recognised'.format(str(device)))
            raise ValueError

        self._owner = owner
        self._device = device
        self._ID = ID
        self._filename = filename
        self._priority = priority
        self._complete = False

        # details relating to function
        self._lib_path = None
        self._module = None
        self._func = None
        self._options = None
        self._params = None

    @property
    def ID(self): return self._ID
    @property
    def device(self): return self._device
    @property
    def owner(self): return self._owner
    @property
    def filename(self): return self._filename
    @property
    def priority(self): return self._priority

    @property
    def job_output(self):
        return self._job_output
    @job_output.setter
    def job_output(self, job_output):
        if not isinstance(job_output, basestring):
            raise TypeError('Output folder must be a string')

        # check and make the directory
        check_and_makedir(job_output)
        self._job_output = job_output


    @staticmethod
    def load(filename, header_only=False):
        return parse_job_file(filename, header_only=header_only)




    @serverlogs.exception_logger
    def __call__(self):
        """ Run the actual method """

        # sanity check - make sure the job is not complete
        if self.complete: return

        # if we have a library path, append it
        if self._lib_path:
            sys.path.append(self._lib_path)
            print sys.path

        # now we need to import job module and get a reference to the function
        m = importlib.import_module(self._module)
        func = getattr(m, self._func)

        # add the output folder to the params
        self._params['output'] = self.job_output

        # call it, with the options and parameters from the job description
        func(self._params, self._options)


    @property
    def complete(self): return self._complete
    @complete.setter
    def complete(self, flag=False):
        """ Set the flag to complete for the job:

        Renames the file as follows:

            JOB_7fab45d.job
            JOB_7fab45d.job.complete

        """
        if not flag: return

        # for good measure, delete the old version
        src = self._filename
        dst = self._filename+'.complete'
        try:
            os.rename(src, dst)
            # print src, dst
        except OSError:
            raise OSError('Failed to set job flags to complete')












def worker(args, log=True):
    """ Worker function.

    Args:
        args.out: this is the output directory for the job. Used for logging
            and output of job related files.
        args.job: the full path to the job description file. This is used to
            configure the job
        log: boolean flag to enable logging

    Returns:
        success: boolean flag

    """

    if not isinstance(args, argparse.Namespace):
        raise TypeError('Args must be of type argsparse.Namespace. The worker'
                        ' function cannot be called directly.')

    # check that the args contains both a job and output directory
    if any([a not in args for a in ['out','job']]):
        raise AttributeError('Could not find .out or .job in args.')

    # start by making the output directory
    check_and_makedir(args.out)

    if log:
        # setup logging here
        import logging
        log_file = serverlogs.setup_logging(args.out, log_name='worker_process')
        logger = logging.getLogger('worker_process')

        logger.info(args.job)

    # load the job and parse the contents
    job = JobWrapper.load(args.job)

    # if we have a properly specified job, run it!
    if job is not None:
        job.job_output = args.out
        job()

    if log: logging.shutdown()
    return




# def SERVER_test(params, options):
#     time.sleep(10)
#     return




if __name__ == '__main__':
    # parse the job file as an input argument and pass to the worker to execute
    # worker returns following the job
    parser = argparse.ArgumentParser(description='ImPy worker process')
    parser.add_argument('--job', help='Path to job description file')
    parser.add_argument('--out', help='Path to output folder')
    args = parser.parse_args()

    # call the function
    worker(args)
