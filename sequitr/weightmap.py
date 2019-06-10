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



class ImageWeightMap2(object):
    """ ImageWeightMap2

    Calculate a per-pixel weight map to prioritise learning of certain pixels
    within an image. Here, the weight map is calculated the distance between
    objects in the foreground for binary images.

    The algorithm proceeds as:
    1. Create a list of xy points that represent the boundaries of the
       foreground objects
    2. Create a Delaunay graph connecting each of the xy points
    3. For each background pixel, calculate the mean length of the edges of
       the simplex in which the pixel lies
    4. Set the pixel of the background to be the mean length value
    5. Calculate an exponential decay of the weight map

    Effectively, the algorithm generates a map of the 'narrowness' of regions
    separating foreground objects. Where objects are separated by only a single
    pixel, the value is high, larger separation decay to zero.

    Params:
        w0: the weighting amplitude
        sigma: the decay of the exponential function

    Notes:
        TODO(arl): clean up the code!
    """
    def __init__(self, w0=10., sigma=5.):
        ImagePipe.__init__(self)
        self.w0 = w0
        self.sigma = sigma

    def __call__(self, image):

        # make a von Neumann structring element to create the boundaries
        s = np.array([[0,1,0],[1,1,1],[0,1,0]])
        b = np.squeeze(image.astype('bool'))
        b_erode_outline = np.logical_xor(binary_erosion(b, iterations=1, structure=s), b)

        # make the sentinels
        b_dilate = binary_dilation(b, iterations=3, structure=s)
        b_dilate_outline = np.logical_xor(binary_erosion(b_dilate, iterations=1, structure=s), b_dilate)

        # add a perimeter of ones to make sentinel points for the boundaries
        b_erode = np.logical_xor(b_erode_outline, b_dilate_outline)

        # pre weight the mask using only the region surrounding the cells
        mask = np.logical_xor(b, b_dilate)

        # assign xy points to the boundary pixels, then a Delaunay triangulation
        x,y = np.where(b_erode)
        points = np.column_stack((x,y))
        tri = Delaunay(points)
        self.tri = tri

        # find the pixels of the background
        free_space_x, free_space_y = np.where(np.logical_not(b))
        free_space = np.array(zip(free_space_x.tolist(), free_space_y.tolist()))

        # calculate the weight map
        simplices = tri.find_simplex(free_space)
        weight_map = np.zeros(image.shape)

        # mean?
        weight_map[free_space_x, free_space_y,...] = np.array([np.max(self.edist(s,p)) for s,p in zip(simplices, free_space)]).reshape((-1,1))

        mask = b[...,np.newaxis].astype('float32')
        weight_map = gaussian_filter(weight_map, 1.) #self.sigma)
        weight_map = self.w0 * (1.-mask) * np.exp(- (weight_map*weight_map) / (2.*self.sigma**2+1e-99) )

        weight_map = weight_map + 1. + mask

        return weight_map

    def edist(self, i, pt):
        if i == -1: return [1024.,1024.,1024.]
        s = self.tri.simplices[i]
        p = np.zeros((4,2))
        # p = np.zeros((3,2))
        p[0:3,:] = self.tri.points[s]
        p[3,:] = p[0,:]

        # d = p - np.tile(pt,(3,1))
        d = np.diff(p, axis=0)
        d = np.sqrt(d[:,0]**2+d[:,1]**2)
        return d

    def _tri_area(self, edist):
        """ Heron's formula..."""
        s = np.sum(edist) / 2.
        return np.sqrt(s*(s-edist[0])*(s-edist[1])*(s-edist[2]))



def create_weightmaps(path,
                      folders,
                      w0=10.,
                      sigma=3.,
                      thresh_fn=lambda x:x>0,
                      name_weights_folder=True):

    """ Generate weightmaps for the images using the binary masks """

    # set up some pipelines
    w_pipe = ImageWeightMap2(w0=w0, sigma=sigma)

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
    create_weightmaps(args.workdir, args.folders, w0=args.w0, sigma=args.sigma)
