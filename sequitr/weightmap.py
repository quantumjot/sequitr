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

__author__ = "Alan R. Lowe"
__email__ = "code@arlowe.co.uk"

import os
import re
from dataio import tifffile as t
from pipeline import ImagePipeline, ImageWeightMap2
import utils
import numpy as np



class ImageLabels(object):
    """ ImageLabels

    A class to deal with image labels.
    """
    def __init__(self,
                 filename,
                 thresh_fn=lambda x:x>0,):
        self._raw_data = t.imread(filename)
        print self._raw_data.shape

        # make sure we have a reasonable number of dimensions
        assert(self._raw_data.ndim > 1 and self._raw_data.ndim < 4)

        # preprocess the data here
        if self._raw_data.ndim == 3:
            l_data = np.zeros(self._raw_data.shape[1:], dtype='uint8')
            for l in range(self._raw_data.shape[0]):
                l_data[thresh_fn(self._raw_data[l,...])] = l+1
                raw_labels = range(self._raw_data.shape[0]+1)
        else:
            l_data = thresh_fn(self._raw_data).astype('uint8')
            raw_labels = [0, 1]

        # convert the label file into an unpacked version? no, but we may
        # need to change so that the labels are 0,1,2...
        # raw_labels = np.unique(l_data)
        self._outputs = len(raw_labels)

        if self.outputs > 5:
            raise ValueError('More that five output classes!')

        print 'Compressing labels from {0:s} to {1:s}'.format(np.unique(l_data), str(raw_labels))
        self._labels = l_data


    def labels(self):
        """ return the labels """
        return self._labels

    @property
    def outputs(self):
        return self._outputs




# import matplotlib.pyplot as plt

def create_weightmaps(path,
                      folders,
                      w0=10.,
                      sigma=3.,
                      thresh_fn=lambda x:x>0,
                      name_weights_folder=True):

    """ Generate weightmaps for the images using the binary masks """

    # set up some pipelines
    w_pipe = ImagePipeline()
    w_pipe.pipeline = [ImageWeightMap2(w0=w0, sigma=sigma)]


    for d in folders:
        r_dir = os.path.join(path, d)
        f_labels = os.listdir(os.path.join(r_dir,'label/'))
        f_labels = [l for l in f_labels if l.endswith('.tif')]

        w_dir_base = 'weights'
        if name_weights_folder:
            w_dir_base += '_w0-{0:2.2f}_sigma-{1:2.2f}'.format(w0, sigma)

        w_dir = os.path.join(r_dir, w_dir_base)
        utils.check_and_makedir(w_dir)

        for f in f_labels:
            print 'Calculating weights for {0:s} in folder \'{1:s}\''.format(f,d)

            w_label = re.match('([a-zA-Z0-9()]+)_([a-zA-Z0-9()]+_)*', f).group(0)
            w_label += 'weights.tif'

            # im_label = t.imread(os.path.join(r_dir,'label/',f))
            # im_label = thresh_fn(im_label)
            label_filename = os.path.join(r_dir,'label/',f)
            im_label = ImageLabels(label_filename).labels()
            im_weights = np.squeeze(w_pipe(im_label.astype('bool')))

            t.imsave(os.path.join(w_dir, w_label), im_weights.astype('float32'))




if __name__ == '__main__':
    import argparse

    DEFAULT_WORKDIR = "/media/lowe-sn00/TrainingData/"

    p = argparse.ArgumentParser(description='Sequitr: weightmap calculation')
    p.add_argument('-p','--workdir', default=DEFAULT_WORKDIR,
                    help='Path to the image data')
    p.add_argument('-f', '--folders', nargs='+', required=True,
                    help='Specify the sub-folders of image data')
    p.add_argument('--w0', type=float, default=30.,
                    help='Specify the amplitude')
    p.add_argument('--sigma', type=float, default=3.,
                    help='Specify the sigma')


    args = p.parse_args()

    print args

    # path = '/media/lowe-sn00/TrainingData/competition_fCNN/'
    # folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
    # print folders
    create_weightmaps(args.workdir, args.folders, w0=args.w0, sigma=arg.sigma)
