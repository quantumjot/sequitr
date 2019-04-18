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
import sys
import numpy as np
import random
import inspect
import json

from skimage.transform import rotate, resize
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from collections import OrderedDict




class ImagePipeline(object):
    """ ImagePipeline

    Method to chain together (pipe) image processing operations to simplify
    uniform pre-processing of images for training and testing.

    Properties:
        pipeline: a python list of ImagePipe objects

    Methods:
        __call__: return the output of the pipe from the given input
        __len__: return the number of images resulting from the pipeline
        update: update each ImagePipe method
        save: save a JSON file with the pipeline parameters
        load: load a JSON file with the pipeline parameters

    Notes:
        None

    """
    def __init__(self, pipeline=[]):
        self.pipeline = pipeline

    @property
    def pipeline(self): return self.__pipeline
    @pipeline.setter
    def pipeline(self, pipeline):
        if isinstance(pipeline, list):
            if pipeline and not any([isinstance(p, ImagePipe) for p in pipeline]):
                raise TypeError('Pipeline contains non pipe objects')
            self.__pipeline = pipeline

    def __call__(self, image):
        """ Feed the image through the pipeline """
        for pipe in self.pipeline:
            image = pipe(image)
        return image

    def __len__(self):
        """ Return the number of images generated by the pipeline """
        n_images = 1
        for pipe in self.pipeline:
            n_images = n_images * len(pipe)
        return n_images

    def update(self):
        """ Run the update of all pipes in the pipeline """
        for pipe in self.pipeline: pipe.update()

    def save(self, filename):
        """ Write out the pipeline as a JSON file """
        save_image_pipeline(filename, self)

    @staticmethod
    def load(filename):
        """ Read in the pipeline from a JSON file """
        return load_image_pipeline(filename)





def save_image_pipeline(filename, pipeline_object):
    """ Save out the parameters of an ImagePipeline as a JSON file.

    Args:
        filename: filename (including path) to the file.
        pipeline: an instance of an ImagePipeline

    Notes:
        TODO(arl): take care of default argument values
    """

    if not isinstance(pipeline_object, ImagePipeline):
        raise TypeError('Pipeline must be of type ImagePipeline')

    if not filename.endswith('.json'):
        filename+='.json'

    pipes = []

    for pipe in pipeline_object.pipeline:
        # write out the pipe and the parameters
        pipe_args = inspect.getargspec(pipe.__init__)[0]
        pipe_args.pop(0) # remove 'self'
        pipe_vals = {p:getattr(pipe,p) for p in pipe_args}
        pipes.append((pipe.__class__.__name__, pipe_vals))

    portal = {'ImagePipeline': OrderedDict(pipes)}

    with open(filename, 'w') as json_file:
        json.dump(portal, json_file, indent=2, separators=(',', ': '))

        # logging.info('Written out ImagePipeline: {0:s}'.format(filename))

def load_image_pipeline(filename):
    """ Load and create an ImagePipeline object from a file.

    Notes:
        Currently this doesn't do any error checking and could be unsafe...
    """

    with open(filename, 'r') as json_file:
        pipes = json.load(json_file, object_pairs_hook=OrderedDict)

    pipeline = []

    # traverse the root node and setup the appropriate pipes
    for p in pipes['ImagePipeline']:
        Pipe = getattr(sys.modules[__name__], p)
        pipeline.append( Pipe(**pipes['ImagePipeline'][p]) )

    return ImagePipeline(pipeline)







class ImagePipe(object):
    """ ImagePipe

    Primitive image pipe. These can be chained together to perform repetitive
    image manipulation tasks.

    Note:
        The ImagePipe is a base class which must be subclassed to function.
    """
    def __init__(self):
        self.iter = 0

    def __call__(self, image):

        # if not isinstance(image, core.Image):
        #     raise TypeError('image must be of type core.Image')
        if image.ndim < 3:
            image = image[...,np.newaxis].astype('float32')
        return self.pipe(image)

    def pipe(self, image):
        raise NotImplementedError('Image pipe is not defined.')

    def __len__(self):
        return 1

    def update(self):
        self.iter = (self.iter+1) % len(self)





class ImageResize(ImagePipe):
    """ ImageResize

    Resize an image to required dimensions. Can use higher order interpolation
    to smooth images. Order should be zero for binary images to preserve hard
    edges.

    Params:
        size: the desired output size as a tuple
        order: the interpolation order
    """
    def __init__(self, size=(1024,1024), order=0):
        ImagePipe.__init__(self)
        self.size = size
        self.order = order

    def pipe(self, image):
        # need to scale range of image for resize
        min_raw = np.min(image)
        max_raw = np.max(image)
        image = (image - min_raw) / (max_raw - min_raw)

        # set the axes for skimage
        image = resize(image, self.size, order=self.order)

        # now scale it back
        return image * (max_raw-min_raw) + min_raw




class ImageFlip(ImagePipe):
    """ ImageFlip

    Perform mirror flips in sequence. Used for data augmentation purposes.
    """
    def __init__(self):
        ImagePipe.__init__(self)
        self.flips = [[],[np.fliplr],[np.flipud],[np.fliplr, np.flipud]]

    def pipe(self, image):
        for flip in self.flips[self.iter]:
            image = flip(image)
        return image

    def __len__(self):
        return len(self.flips)


class ImageBlur(ImagePipe):
    """ ImageBlur

    Perform a Gaussian filtering of the input image. All filtering occurs in 2D
    on each successive layer of the image.

    Params:
        sigma: the standard deviation of the symmetrical 2D Gaussian filter
    """
    def __init__(self, sigma=0.5):
        ImagePipe.__init__(self)
        self.sigma = sigma

    def pipe(self, image):
        for chnl in xrange(image.shape[-1]):
            image[...,chnl] = gaussian_filter(image[...,chnl], self.sigma)
        return image

    def __len__(self):
        return len(self.flips)


class ImageOutliers(ImagePipe):
    """ ImageOutliers

    Remove outliers from an image, for example hot pixels on a CMOS or CCD
    camera. Works by calculating a median filtered version of the image (radius
    2 pixels) and compares this with the raw image.  Where there is a
    significant difference between the raw and filtered, the pixel in the raw
    image is replaced by the median value.

    Args:
        image: takes a standard 2d image (or numpy array)
        threshold: difference between the median filtered and raw image

    Returns:
        image with hot-pixels removed.

    """
    def __init__(self, sigma=2, threshold=5.):
        ImagePipe.__init__(self)
        self.sigma = sigma
        self.threshold = threshold

    def pipe(self, image):
        for chnl in xrange(image.shape[-1]):
            filtered_image = image[...,chnl].copy()
            med_filtered_image = median_filter(filtered_image, self.sigma)
            diff_image = np.abs(image[...,chnl] - med_filtered_image)
            differences = diff_image>self.threshold
            filtered_image[differences] = med_filtered_image[differences]
            image[...,chnl] = filtered_image
        return image


class ImageRotate(ImagePipe):
    """ ImageRotate

    Rotate an image by a certain angle in degrees. Boundaries are reflected, but
    this can cause some issues with objects at the periphery of the FOV.

    Args:
        rotations: the number of rotations to perform
        order: the interpolation order, important when dealing with binary image
        max_theta: the angle over which to rotate

    Notes:
        None
    """
    def __init__(self, rotations=16, order=0, max_theta=360):
        ImagePipe.__init__(self)
        self.rotations = rotations
        self.max_theta = max_theta
        self.order = order
        self.iter = 0

    @property
    def theta(self):
        return (-self.max_theta/2.) + self.max_theta * (float(self.iter) / \
            float(self.rotations))

    def pipe(self, image):
        r = [np.min(image), np.max(image)]
        image = (image-r[0]) / (r[1]-r[0])
        image = rotate(image, self.theta, order=self.order, mode='reflect')
        image = image * (r[1]-r[0]) + r[0]
        return image

    def __len__(self):
        return self.rotations




class ImageNorm(ImagePipe):
    """ ImageNorm

    Normalise an image by subtracting the mean and dividing by the standard
    deviation. This should return an image with a mean of zero and a standard
    deviation of one.

    Notes:
        None
    """
    def __init__(self):
        ImagePipe.__init__(self)
        self.epsilon = 1e-99

    def pipe(self, image):
        for chnl in xrange(image.shape[-1]):
            image[...,chnl] = (image[...,chnl] - np.mean(image[...,chnl])) / \
                (self.epsilon+np.std(image[...,chnl]))
        return image



class ImageBGSubtract(ImagePipe):

    """ estimate_background

    Estimate the background of an image using a second-order polynomial surface
    assuming sparse signal in the image.  Essentially a massive least-squares fit of the
    image to the polynomial.

    Args:
    	image -	An input image which is to be used for estimating the background.

    Returns:
    	A second order polynomial surface representing the estimated background of the image.

    Notes:
        Old slow looping code now replaced by fast numpy matrix code

    """
    def __init__(self):
        ImagePipe.__init__(self)

    def pipe(self, image):
        w,h, channels = image.shape

        # set up arrays for params and the output surface
        A = np.array(np.zeros((image.shape[0]*image.shape[1],6)))
        background_estimate = np.array(np.zeros((image.shape[1],image.shape[0])))

        u, v = np.meshgrid(np.arange(0,image.shape[1]), np.arange(0,image.shape[0]))
        A[:,0] = 1.
        A[:,1] = np.reshape(u,(image.shape[0]*image.shape[1],))
        A[:,2] = np.reshape(v,(image.shape[0]*image.shape[1],))
        A[:,3] = A[:,1]**2
        A[:,4] = A[:,1]*A[:,2]
        A[:,5] = A[:,2]**2

        # convert to a matrix
        A = np.matrix(A)

        # calculate the parameters
        k = np.linalg.inv(A.T*A)*A.T
        k = np.array(np.dot(k,np.ravel(image)))[0]

        # calculate the surface
        background_estimate = k[0] + k[1]*u + k[2]*v + k[3]*(u**2) + k[4]*u*v + k[5]*(v**2)
        return image - background_estimate[...,np.newaxis]


class ImageSample(ImagePipe):
    """ ImageSample

    Sample an image by extracting randomly selected Regions of Interest (ROI).
    A number of samples can be taken, limited by the size of the image. The
    positions of each ROI are stored so that the same regions of corresponding
    labels or weights can be selected.

    Args:
        samples: number of samples to take
        ROI_size: the size as a tuple (in pixels) of the ROI to crop

    """
    def __init__(self, samples=16, ROI_size=(512,512)):
        ImagePipe.__init__(self)
        self.samples = samples
        self.ROI_size = ROI_size
        self.im_size = None
        self.boundary = int(ROI_size[0] / 2.)
        self.coords = None

    def pipe(self, image):
        sampled = np.zeros((self.samples, self.ROI_size[0], self.ROI_size[1],
            image.shape[-1] ))
        self.im_size = image.shape

        if not self.coords: self.update()

        for sample, (x,y) in enumerate(self.coords):
            # create some random samplings of the image
            sampled[sample,...] = image[x-self.boundary:x+self.boundary,
                y-self.boundary:y+self.boundary,...]

        return sampled

    def update(self):
        x = np.random.randint(self.boundary, high=self.im_size[0]-self.boundary,
            size=(self.samples,))
        y = np.random.randint(self.boundary, high=self.im_size[1]-self.boundary,
            size=(self.samples,))
        self.coords = zip(x,y)

    def __len__(self):
        return self.samples



class ImageWeightMap(ImagePipe):
    """ ImageWeightMap

    Calculate a per-pixel weight map to prioritise learning of certain pixels
    within an image. Here, the weight map is calculated as an exponential decay
    away from the edges of binary objects.

    All objects have a minimum weighting of 1.0, with higher weightings applied
    to the pixels in the background near the edges of foreground. The parameters
    w0 and sigma control the amplitude and decay of the weighting.

    Params:
        w0: the weighting amplitude
        sigma: the decay of the exponential function
    """
    def __init__(self, w0=10., sigma=5.):
        ImagePipe.__init__(self)
        self.w0 = w0
        self.sigma = sigma

    def pipe(self, image):
        weight_map = distance_transform_edt(1.-image)
        weight_map = self.w0 * (1.-image) * np.exp(- (weight_map*weight_map) / \
        (2.*self.sigma**2+1e-99) )
        return  weight_map + image + 1.


class ImageWeightMap2(ImagePipe):
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

    def pipe(self, image):
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










if __name__ == '__main__':
    pass
    # p = ImagePipeline([ImageNorm(), ImageMaskEdge(), ImageFlip()])
    # p.save('/home/arl/Documents/Pipe.test.json')
    # p.load('/home/arl/Documents/Pipe.test.json')
