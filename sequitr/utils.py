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

import re
import os
import h5py
import random
import json
import logging
import itertools
import inspect
import functools

import numpy as np

from shutil import copyfile
from collections import OrderedDict

import tensorflow as tf

# import impy core for globals
import core


# get the logger instance
logger = logging.getLogger('worker_process')

# some wrappers for device placement
def network_device_placement(device):
    """ A simple decorator to aid in placement of larger networks.

    Wraps a function which adds nodes to a graph with a device context
    manager, to enable easy placement of that graph.

    Use:
        @network_device_placement("/gpu:0")
        def build_graph(x, y, z):
            pass

    """
    assert(device in ("/gpu:0","/gpu:1","/gpu:2","/gpu:3"))
    def wrapper(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            print 'Building {0:s} on dev:{1:s}'.format(repr(function), device)
            with tf.device(device):
                return function(*args, **kwargs)
        return wrapped
    return wrapper

# some wrappers for device placement
def as_tf_session(**tf_session_kwargs):
    """ A simple decorator to wrap functions in a TensorFlow session.
    """
    def wrapper(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            session = tf.Session(**tf_session_kwargs)
            kwargs['session'] = session
            with session.as_default():
                return function(*args, **kwargs)
        return wrapped
    return wrapper



def filter_doubling(start_filters=8,
                    num_layers=7,
                    max_filters=4096,
                    reverse=False):
    """ Set up a sequence of filters, doubling at every level

    Args:
        start_filters:  (int) the number of filters in the first level
        num_layers:     (int) the number of layers/filters to generate
        max_filters:    (int) set an upper limit on the number of filters
        reverse:        (bool) reverse the sequence to descending

    Returns:
        filters:        a list of filters, e.g. [4, 8, 16, 32, 64]

    """
    f = [min(start_filters*(2**i), max_filters) for i in range(num_layers)]
    if reverse: f.reverse()
    return f




def check_and_makedir(folder_name):
    """ Does a directory exist? if not create it. """
    if not os.path.isdir(folder_name):
    	logger.info('Creating output folder {0:s}...'.format(folder_name))
    	os.mkdir(folder_name)
    	return False
    else:
    	return True


def get_latest_model_dir(export_dir_base):
    """ Return the folder containing the most recent model """

    # now check if there are any existing folders
    models = [f for f in os.listdir(export_dir_base) if os.path.isdir(os.path.join(export_dir_base, f))]

    export_dir_fn = lambda x: '{0:d}'.format(x).zfill(4)

    if not models:
        return None
    else:
        model_filenumbers = [int(f) for f in models]
        latest_model = max(model_filenumbers)
        return os.path.join(export_dir_base, export_dir_fn(latest_model))


def create_new_export_dir(export_dir_base):
    """ Create a numbered model export directory """

    # first make the export dir if necessary
    check_and_makedir(export_dir_base)

    export_dir_fn = lambda x: '{0:d}'.format(x).zfill(4)

    # get the most recent model directory and increment
    latest_model = get_latest_model_dir(export_dir_base)
    if latest_model is None:
        model_num = export_dir_fn(0)
    else:
        _, model_num = os.path.split(latest_model)

    # create the new model folder
    new_dir = export_dir_fn(int(model_num)+1)
    new_dir = os.path.join(export_dir_base, new_dir)

    # make a new directory
    if check_and_makedir(new_dir):
        raise IOError('New export model dir already exists?!?!')
    return new_dir




def save_estimator_model(estimator, config):
    """ Update the stored estimator model. This is a bit of a hack but allows
    a net to be reconfigured after training.

    model_folder/
        0001/
        0002/
            checkpoint

    """

    if not isinstance(config, NetConfiguration):
        raise TypeError('Configurations needs to be of type NetConfiguration')

    # now copy over the checkpoint to the models directory
    latest_checkpoint = estimator.latest_checkpoint()
    checkpoint_dir, checkpoint = os.path.split(latest_checkpoint)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(checkpoint)]

    # make an output model folder in the model folder
    export_dir = create_new_export_dir(config.export_dir_base)

    logger.info('Saving estimator model: {0:s}'.format(export_dir))
    logger.info('Checkpoint dir: {0:s}'.format(checkpoint_dir))

    # also add the checkpoint file
    checkpoint_files += ['checkpoint']

    # copy the files over
    for f in checkpoint_files:
        src = os.path.join(checkpoint_dir, f)
        dst = os.path.join(export_dir, f)
        copyfile(src, dst)

    # save a copy of the configuration file
    logger.info('Saving net configuration...')
    config.save(os.path.join(export_dir, 'net.config'))

    # logger.info('Updated model folder: {0:s}...'.format(export_dir))




def power_of_two(number):
    """ Return a bool testing whether number is power of two or not """
    return int(bin(number & (number-1)),2) == 0

def divisible_by_two_n_times(x, n):
    """ Check whether a number is divisible by two, n times. Useful for
    checking whether the dimensions of an image are compatibile with multiple
    max_pool levels. """
    for i in range(n):
        x = x/2.0
    return x % 1 == 0






class NetConfiguration(object):
    """ NetConfiguration

    A generic configuration for a network class. Several of these parameters can
    be overwritten when specifying a job from the server.


    Params:
        name
        filters
        dropout
        num_inputs
        num_outputs
        num_epochs
        augment
        batch_size
        learning_rate
        warm_start
        shape
        path
        training_data
        image_dict

    Methods:
        warm_start_from: return the path to a previous model checkpoint
        from_params: create a configuration from a parameter dict
        to_params: create a parameter dictionary from the configuration
        save:
        load:
        get_latest_model_dir:


    """

    def __init__(self):
        self.name = 'UNet2D_test'
        self.dropout = 0.5
        self.warm_start = False
        self.shape = (64,64)
        self.num_inputs = 1
        self.num_outputs = 2
        self.num_epochs = 1000
        self.learning_rate = 0.01
        self.augment = True
        self.path = None
        self.training_data = 'test.tfrecord'
        self.image_dict = {}

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, name):
        if not isinstance(name, basestring):
            raise TypeError('Name is not a string.')
        if name not in MODELS:
            raise ValueError('Net name is not recognized.')
        self._name = name

    @property
    def dropout(self):
        return self._name
    @dropout.setter
    def dropout(self, dropout):
        if not isinstance(dropout, float):
            raise TypeError('Dropout is not a float.')
        if dropout < 0 or dropout > 1:
            raise ValueError('Dropout should be in the (0-1) range.')

    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, shape):
        if not isinstance(shape, (tuple,list)):
            raise TypeError('Shape is not a tuple.')
        #TODO(arl): check for decent limits on this
        self._shape = shape

    @property
    def warm_start(self):
        return self._warm_start
    @warm_start.setter
    def warm_start(self, warm_start):
        if not isinstance(warm_start, bool):
            raise TypeError('Warm start is not a boolean.')
        self._warm_start = warm_start

    @property
    def export_dir_base(self):
        MODELDIR = core.TensorflowConfiguration.MODELDIR
        return os.path.join(MODELDIR, self.name)

    def warm_start_from(self, model_num=None):
        """ Get the latest version of the model """
        #TODO(arl): allow warm start from arbitrary model number
        if not self.warm_start: return None
        return get_latest_model_dir(self.export_dir_base)

    def get_latest_model_dir(self):
        return get_latest_model_dir(self.export_dir_base)

    @property
    def training_data_file(self):
        if isinstance(self.training_data, list):
            return [os.path.join(self.path, f) for f in self.training_data]
        return os.path.join(self.path, self.training_data)

    @classmethod
    def from_params(cls, params, preload_model=False):
        """ Return an instantiated config from a parameter dictionary.

        NOTE(arl): The load flag will preload model details from a saved file,
        any further options specified by the params dictionary will overwrite
        these. This is meant to preserve parameters of the network between
        training and inference.
        """

        if not isinstance(params, dict):
            raise TypeError('Parameters are not specified in dictionary.')

        config = cls()

        # load model params if found
        if preload_model:
            config.name = params['name']
            config.load()

        # TODO(arl): this is lazy - make sure we're adding members that exist
        print config
        for p in params:
            setattr(config, p, params[p])
            print "'{0:s}': {1:s}".format(p, str(getattr(config, p)))

        return config

    def to_params(self):
        """ Return the config as a list of parameters """
        # TODO(arl): this needs to be cleaned up
        members = [m.lstrip('_') for m in self.__dict__.keys()]
        params = {m:getattr(self, m) for m in members}
        return params

    def save(self, filename):
        """ Export the config to a file """
        # TODO(arl): some error checking on the filename
        export = {str(self.__class__.__name__): self.to_params()}
        with open(filename, 'w') as f:
            f.write(json.dumps(export, indent=2, separators=(',', ': ') ))


    def load(self, filename='net.config'):
        """ Import the config from a file """
        model_dir = self.get_latest_model_dir()
        model_fn = os.path.join(model_dir, filename)
        if not os.path.exists(model_fn):
            raise IOError('Cannot preload config: {0:s}'.format(model_fn))

        with open(model_fn, 'r') as f:
            config = json.load(f)
            params = config[str(self.__class__.__name__)]

            logger.info('Loading model parameters from: {0:s}'.format(model_fn))
            for p in params:
                setattr(self, p, params[p])
                print "'{0:s}': {1:s}".format(p, str(getattr(self, p)))

        return









class HDF5FileHandler(object):
    """ Base class for handling HDF files """

    def __init__(self, filename=None, read_only=False):

        # check that we've specified a string for the filename
        if not isinstance(filename, basestring):
            TypeError('Filename must be specified as a string')

        # check that the destination path exists
        pth, f = os.path.split(filename)
        if not os.path.exists(pth):
            raise IOError('Destination path {0:s} doesn\'t exist'.format(pth))

        # make sure that we have the correct extension
        f_noext, f_ext = os.path.splitext(f)
        if f_ext != '.hdf5':
            filename = f_noext + '.hdf5'

        self.filename = filename


        # set whether this has read or write access
        if read_only:
            read_write_flag = 'r+'
        else:
            read_write_flag = 'w' #w

        # set the read only flag
        self.read_only = read_only

        logger.info('Opening HDF file: {0:s}'.format(filename))
        self._hdf = h5py.File(filename, read_write_flag)

    @property
    def hdf(self): return self._hdf

    def __del__(self):
        if self._hdf: self.close()

    def close(self):
        """ Manually close the HDF5 file, to prevent HDF5 corruption """
        logger.info('Closing HDF file.')
        self._hdf.close()




class CentroidWriter(HDF5FileHandler):
    """ CentroidWriter

    Using the segmentation output, find the centre of mass of each object
    and write these to the HDF file, each group of centroids is grouped by the
    frame in which they were found.

    This works with both images (2D) and volumes (3D).

    """
    def __init__(self, filename=None):
        HDF5FileHandler.__init__(self, filename)
        # set up the basic structure
        self._hdf.create_group('frames')


    def write(self, segmented):
        """ Take a (large!) numpy array and output dataset """

        # import the labeling and centroid finding functions
        from scipy.ndimage.measurements import label, center_of_mass


        if segmented.ndim == 4:
            # volumetric
            def get_cartesian_coords(coords):
                x,y,z = zip(*coords)
                return x,y,z
            im_type = "Volumetric"

            # also need to swap some axes, default input is N,Z,X,Y
            segmented = np.swapaxes(segmented, 1,-1)
            print segmented.shape
        elif segmented.ndim == 3:
            # planar
            def get_cartesian_coords(coords):
                x,y = zip(*coords)
                return x,y,[0.0]*len(x)
            im_type = "Image"
        else:
            logger.error("Incorrect image data shape.")

        for i in xrange(segmented.shape[0]):
            if i % 100 == 0:
                logger.info('Written out {0:d} of {1:d} frames ({2:s})...'
                            .format(i, segmented.shape[0], im_type))


            # if we have multiple classes in the segmentation output, gather
            # them here, then make a separate list of centroids for each...
            out = segmented[i,...]
            classes = [x for x in np.unique(out) if x>0]

            this_frame = []

            for c in classes:

                # extract centroids
                matrix, n_labels = label(out == c)
                labels = [l for l in np.unique(matrix) if l>0]

                coords = center_of_mass(out, matrix, labels)

                # if we don't find anything, skip this frame
                if len(coords) < 1: continue

                # use the correct method to get the volumetric/planar coords
                x,y,z = get_cartesian_coords(coords)

                # make a numpy array
                this_class = np.zeros((len(x),5), dtype='float32')
                this_class[:,0] = i
                this_class[:,1] = x
                this_class[:,2] = y
                this_class[:,3] = z # z is zero by default
                this_class[:,4] = c

                this_frame.append(this_class)


            # # add (or append) the coordinates to the dataset
            grp = self._hdf['frames'].create_group('frame_'+str(i))

            if this_frame:
                # concatenate into one array
                this_frame = np.concatenate(this_frame, axis=0)

            grp.create_dataset('coords',
                                data=this_frame,
                                dtype='float32')





def get_model_list():
    """ Return the list of models """
    with open("./models.txt",'r') as models_file:
        models_raw = models_file.readlines()
    models = [m.rstrip('\r\n') for m in models_raw]
    models = filter(None, models)
    return tuple(models)

# set up the list of models
MODELS = get_model_list()










def batch_rename(path, stem, new_stem):
    """
        Rename images from ImageJ
    """
    files = os.listdir(path)
    for file in files:
        filename = re.match('('+stem+')([0-9]*)(\.tif)', file)
        if filename:
            old_filename = os.path.join(path,file)
            new_filename = os.path.join(path,filename.group(2)+new_stem+filename.group(3))
            os.rename( old_filename, new_filename )
            print old_filename, new_filename
    return True







if __name__ == '__main__':
    pass
