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

import os
import core
import utils
import logging

import numpy as np
import tensorflow as tf

# set verbose logging
tf.logging.set_verbosity(tf.logging.INFO)


LOGDIR = core.TensorflowConfiguration.LOGDIR
MODELDIR = core.TensorflowConfiguration.MODELDIR
UNET_MODEL_FOLDER = MODELDIR



DEFAULT_FILTERS = (16, 32, 64, 128, 256)
DEFAULT_DROPOUT = 0.4
BRIDGE_TYPES = ('eltwise_add', 'eltwise_mul', 'eltwise_sub', 'concat', None)


# get the logger instance
logger = logging.getLogger('worker_process')






class UNet(object):
    """ UNet

    ** This is the Base Class, use the sublasses UNet2D or UNet3D **

    A UNet class for image segmentation, implemented using TensorFlow.
    Basic architechture nomenclature used here: L0u (Layer 0, up)
        - Bridge
        - Layers a labeled from the top (0) to the bottom, e.g. 4
        - Layers are labeled as up or down

    This implementation differs in that we pad each convolution such
    that the output following convolution is the same size as the input.
    Also, bridges are elementwise operations of the filters to approach a
    residual-net architecture (resnet), although this can be changed by
    the user.  The bridge_type property allows different bridge types to be
    specified:
        - elementwise_add
        - elementwise_multiply
        - elementwise_subtract
        - concatenate
        - None (no bridge information, resembles an autoencoder)

    Image autoencoders can also be subclassed from this structure, by
    removing the bridge information.

    Note that the UNet class should not be used on it's own. Generally
    there are subclassed versions which inherit the main features but
    specify loss functions and bridge details that are specific to the
    particular architecture.

    TODO(arl): implement filter doubling

    Args:
        params: a network configuration object dict (usually from
            utils.NetConfiguration)
        mode: the tensorflow training mode flag

    Properties:
        _activation: the activation function to use, e.g. tf.nn.relu
        _initializer: default initializer for kernels

        name: a name for the network, this should be on the white-listed network
            names list in the core module

        bridge: name of bridge type ('eltwise_add', 'eltwise_mul', 'concat')
        dropout: dropout rate (e.g. 0.5 during training)

        use_filter_doubling: (bool) doubles filters within a layer before the
            maxpool/conv_transpose layers to prevent bottlenecks

    Methods:
        build():    build the network

    Notes:
        Based on the original publications:

        U-Net: Convolutional Networks for Biomedical Image Segmentation
        Olaf Ronneberger, Philipp Fischer and Thomas Brox
        http://arxiv.org/abs/1505.04597

        3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
        Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox
        and Olaf Ronneberger
        https://arxiv.org/abs/1606.06650

        Filter doubling from:
        Rethinking the Inception Architecture for Computer Vision.
        Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojn, Z.
        https://arxiv.org/abs/1512.00567


    """
    def __init__(self, params, mode):

        # training mode: TRAIN, TEST or PREDICT
        self._mode = mode

        # TODO(arl): proper error checking on these parameters
        self.name = params.get('name','UNet2d_test')
        self.filters = params.get('filters', DEFAULT_FILTERS)
        self.dropout = params.get('dropout', DEFAULT_DROPOUT)
        self.n_inputs = params.get('num_inputs', 1)
        self.n_outputs = params.get('num_outputs', 2)
        self.shape = params.get('shape', (1024, 1024))
        self.bridge_type = params.get('bridge', 'eltwise_mul')
        self.kernel = params.get('kernel', (3,3))

        # activation functions and initializers
        self._activation = tf.nn.relu
        self._initializer = tf.initializers.variance_scaling

        # empty net to start with
        self._net = None

    @property
    def width(self):
        """ width of the image volume """
        return self.shape[0]

    @property
    def height(self):
        """ height of the image volume """
        return self.shape[1]

    @property
    def slices(self):
        """ depth (number of slices) of the image volume """
        if self.ndim < 3: return 0
        return self.shape[2]

    @property
    def ndim(self):
        """ number of dimensions of image volume """
        return len(self.shape)

    @property
    def training(self):
        """ training mode flag """
        return self._mode==tf.estimator.ModeKeys.TRAIN

    @property
    def btype(self):
        """ DEPRECATED: bridge type  """
        raise DeprecationWarning("Use @bridge_type")

    @property
    def bridge_type(self):
        return self._bridge_type
    @bridge_type.setter
    def bridge_type(self, bridge):
        """ Set the bridge type """

        if bridge not in BRIDGE_TYPES:
            raise ValueError('Bridge type not recognized')

        # set the bridge function
        if bridge == 'eltwise_add':
            self.bridge = lambda x,y: tf.add(x,y)
        elif bridge == 'eltwise_mul':
            self.bridge = lambda x,y: tf.multiply(x,y)
        elif bridge =='eltwise_sub':
            self.bridge = lambda x,y: tf.subtract(x,y)
        elif bridge == 'concat':
            self.bridge = lambda x,y: tf.concat([x,y],-1)
        else:
            logger.warning('Bridge function in UNet not recognized')
            self.bridge = lambda x,y: x

        self._bridge_type = bridge


    def reshape_input(self, features):
        """ Reshape the input layer from the dataset features:
        (batch, depth (aka slices), height, width, channels)
        """
        # reshape the data to the correct size
        full_shape = [-1, self.slices, self.width, self.height, self.n_inputs]
        input_shape = [d for d in full_shape if d != 0]
        print full_shape, input_shape
        input_layer = tf.reshape(features,
                                input_shape,
                                name='input_layer')
        return input_layer



    def logits(self):
        """ return the un-normalized logits (i.e. last) layer of the network """
        return self._net[-1]

    def build(self, features):
        """ build

        Build the network using the given parameters and the features. Returns
        the final output layers. Input are the features as a tensor, typically
        from a tensorflow Dataset object.
        """
        # output some details of the net
        logger.info('Building UNet ({0:s})...'.format(self.__class__.__name__))

        with tf.variable_scope('UNet'):
            input_layer = self.reshape_input(features)

            # BUILD THE NET!
            self._net = [self.down_layer(input_layer, self.filters[0], name=0)]

            # do the down layers
            for i, f in enumerate(self.filters[1:]):
                prev_layer = self.max_pool_layer(self._net[-1])
                self._net.append( self.down_layer(prev_layer, f, name=i+1) )

            # now add the up layers
            for i, f in reversed(list(enumerate(self.filters[:-1]))):
                prev_layer = self._net[-1] # layer below
                bridge = self._net[i]      # bridge information
                self._net.append( self.up_layer(prev_layer, f, bridge, name=i) )

            # make an output layer with a 1x1 convolution
            with tf.variable_scope('to_image'):
                logits = self.conv_layer_1x1(self._net[-1], self.n_outputs)

        logger.info('Output layer -> shape {0:s}'.format(str(logits.shape)))

        # append this layer for completeness
        self._net.append(logits)

        logger.info('...Done')

        return logits


    def conv_block(self, x, filters):
        """ convolutional block """

        with tf.variable_scope('conv1'):
            conv1 = self.conv_layer(x, filters)
        with tf.variable_scope('conv2'):
            conv2 = self.conv_layer(conv1, filters)

        # Dropout
        drop = tf.layers.dropout(inputs=conv2,
                                 rate=self.dropout,
                                 training=self.training)
        return drop




    def down_layer(self, x, filters, name=None):
        """ down_layer

        A down layer of the UNet. These are characterised by a series of
        convolution and ReLu operations, followed by a max pool to down
        sample to the next layer. A layer here is defined as 2x
        [3x3 convolution, ReLu]

        Tensor shape is often of the format: NHWC
        """
        logger.info('Down layer -> shape {0:s}'.format(str(x.shape)))

        with tf.variable_scope('down{0:d}'.format(name)):
            out = self.conv_block(x, filters)
        return out


    def up_layer(self, x, filters, bridge, name=None):
        """ up_layer

        These are characterised by a series of convolution and ReLu
        operations, followed by a transpose deconvolution to up sample to the
        next layer.

        Tensor shape is often of the format: NHWC
        """

        logger.info('Up layer -> shape {0:s} (bridge: {1:s})'
                    .format(str(x.shape), self.bridge_type))

        with tf.variable_scope('up{0:d}'.format(name)):
            # scale up the image
            with tf.variable_scope('upscale'):
                upscale = self.conv_transpose_layer(x, filters)

            # now we need to incorporate the filters using the bridge
            with tf.variable_scope('bridge'):
                bridge = self.bridge(upscale, bridge)

            out = self.conv_block(bridge, filters)
        return out



    def conv_layer(self, x, filters):
        """ Convolution layer, conv-relu with padding """
        raise NotImplementedError


    def conv_layer_1x1(self, x, filters):
        """ Return a 1x1 convolution layer """
        raise NotImplementedError


    def conv_transpose_layer(self, x, filters):
        """ Transpose convolution (aka deconvolution) layer """
        raise NotImplementedError

    def pool_layer(self, x):
        """ Max pool operation """
        raise NotImplementedError





def tr_augment(features, params):
    """ Augment the dataset by random cropping, flipping and rotations

    Identical augmentations need to be applied to the labels and any weight
    map. For the weight map, we make a mask where the regions outside of the
    actual image have weights of one - this ensures that we don't set incorrect
    labels outside of the actual image data while augmenting...

    """

    img = features['image']
    label = features['label']
    weights = features['weights']
    image_shape = features['shape']
    height = image_shape[1]
    width = image_shape[2]

    outputs = params.get('num_outputs', 2)


    ch, cw = params.get('shape', (512,512))[0:2]

    # random rotation, crop and flips
    theta = 2.*tf.random_uniform([], dtype=tf.float32)*np.pi

    # rotate the images
    im_rot = tf.contrib.image.rotate(img, theta, interpolation='BILINEAR')
    lbl_rot = tf.contrib.image.rotate(label, theta, interpolation='NEAREST')
    wgt_rot = tf.contrib.image.rotate(weights, theta,  interpolation='BILINEAR')

    # make a mask where the regions outside of the actual image have weights
    # of one - this ensures that we don't set incorrect labels outside of
    # the actual image data while augmenting...
    mask_im = tf.ones(image_shape, dtype=tf.float32)
    wgt_mask = 1.0 - tf.contrib.image.rotate(mask_im, theta,
                                             interpolation='NEAREST')
    wgt_rot = tf.add(wgt_rot, wgt_mask)

    if (ch,cw != height,width):
        rh = tf.random_uniform([], maxval=height-ch, dtype='int32')
        rw = tf.random_uniform([], maxval=width-cw, dtype='int32')

        # crop the image, labels and weights
        img = tf.image.crop_to_bounding_box(im_rot, rh, rw, ch, cw)
        label = tf.image.crop_to_bounding_box(lbl_rot, rh, rw, ch, cw)
        weights = tf.image.crop_to_bounding_box(wgt_rot, rh, rw, ch, cw)

    # now expand the label
    channels = range(5) #TODO(arl): this is UGLY!
    labels = [tf.cast(tf.equal(label, chnl), tf.uint8) for chnl in channels]
    label = tf.concat(labels, axis=-1)[...,:outputs]

    # only return the first two channels...
    return img, {'label':label, 'weights':weights}



def preprocess_norm(features):
    """ normalise images or volumes to mean 0. and std 1.0

    if the image shape is: (Z, H, W, C), rank=4, axes = [0,1,2]
                              (H, W, C), rank=3, axes = [0,1]

    """


    try:
        img = features['image']
    except:
        img = features
    axes = tf.range(tf.rank(img)-1)

    # need to normalise these now
    mean, var = tf.nn.moments(img, axes=axes, keep_dims=True)
    img = tf.nn.batch_normalization(img,
                                    mean,
                                    var,
                                    None,
                                    None,
                                    1e-38,
                                    name='image_normalisation')

    # return the normalized image to the dict
    # features['image'] = img
    return features












class UNet_LEGACY(object):
    """ UNet

    ** This is the Base Class, use the sublasses UNet2D or UNet3D **

    A UNet class for image segmentation, implemented using TensorFlow.
    Basic architechture nomenclature used here: L0u (Layer 0, up)
        - Bridge
        - Layers a labeled from the top (0) to the bottom, e.g. 4
        - Layers are labeled as up or down

    This implementation differs in that we pad each convolution such
    that the output following convolution is the same size as the input.
    Also, bridges are elementwise operations of the filters to approach a
    residual-net architecture (resnet), although this can be changed by
    the user.  The bridge_type property allows different bridge types to be
    specified:
        - elementwise_add
        - elementwise_multiply
        - elementwise_subtract
        - concatenate
        - None (no bridge information, resembles an autoencoder)

    Image autoencoders can also be subclassed from this structure, by
    removing the bridge information.

    Note that the UNet class should not be used on it's own. Generally
    there are subclassed versions which inherit the main features but
    specify loss functions and bridge details that are specific to the
    particular architecture.

    TODO(arl): implement filter doubling

    Args:
        params: a network configuration object dict (usually from
            utils.NetConfiguration)
        mode: the tensorflow training mode flag

    Properties:
        _activation: the activation function to use, e.g. tf.nn.relu
        _initializer: default initializer for kernels

        name: a name for the network, this should be on the white-listed network
            names list in the core module

        bridge: name of bridge type ('eltwise_add', 'eltwise_mul', 'concat')
        dropout: dropout rate (e.g. 0.5 during training)

        use_filter_doubling: (bool) doubles filters within a layer before the
            maxpool/conv_transpose layers to prevent bottlenecks

    Methods:
        build():    build the network

    Notes:
        Based on the original publications:

        U-Net: Convolutional Networks for Biomedical Image Segmentation
        Olaf Ronneberger, Philipp Fischer and Thomas Brox
        http://arxiv.org/abs/1505.04597

        3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
        Ozgun Cicek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox
        and Olaf Ronneberger
        https://arxiv.org/abs/1606.06650

        Filter doubling from:
        Rethinking the Inception Architecture for Computer Vision.
        Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojn, Z.
        https://arxiv.org/abs/1512.00567


    """
    def __init__(self, params, mode):

        # training mode: TRAIN, TEST or PREDICT
        self._mode = mode

        # TODO(arl): proper error checking on these parameters
        self.name = params.get('name','UNet2d_test')
        self.filters = params.get('filters', DEFAULT_FILTERS)
        self.dropout = params.get('dropout', DEFAULT_DROPOUT)
        self.n_inputs = params.get('num_inputs', 1)
        self.n_outputs = params.get('num_outputs', 2)
        self.shape = params.get('shape', (1024, 1024))
        self.use_filter_doubling = params.get('use_filter_doubling', False)
        self.bridge_type = params.get('bridge', 'eltwise_mul')
        self.kernel = params.get('kernel', (3,3))

        # activation functions and initializers
        self._activation = tf.nn.relu
        self._initializer = tf.initializers.variance_scaling

        # empty net to start with
        self._net = None

    @property
    def width(self):
        """ width of the image volume """
        return self.shape[0]

    @property
    def height(self):
        """ height of the image volume """
        return self.shape[1]

    @property
    def slices(self):
        """ depth (number of slices) of the image volume """
        if self.ndim < 3: return 0
        return self.shape[2]

    @property
    def ndim(self):
        """ number of dimensions of image volume """
        return len(self.shape)

    @property
    def training(self):
        """ training mode flag """
        return self._mode==tf.estimator.ModeKeys.TRAIN

    @property
    def btype(self):
        """ DEPRECATED: bridge type  """
        raise DeprecationWarning("Use @bridge_type")

    @property
    def bridge_type(self):
        return self._bridge_type
    @bridge_type.setter
    def bridge_type(self, bridge):
        """ Set the bridge type """

        if bridge not in BRIDGE_TYPES:
            raise ValueError('Bridge type not recognized')

        # set the bridge function
        if bridge == 'eltwise_add':
            self.bridge = lambda x,y: tf.add(x,y)
        elif bridge == 'eltwise_mul':
            self.bridge = lambda x,y: tf.multiply(x,y)
        elif bridge =='eltwise_sub':
            self.bridge = lambda x,y: tf.subtract(x,y)
        elif bridge == 'concat':
            self.bridge = lambda x,y: tf.concat([x,y],-1)
        else:
            logger.warning('Bridge function in UNet not recognized')
            self.bridge = lambda x,y: x

        self._bridge_type = bridge


    @property
    def use_filter_doubling(self):
        return self._use_filter_doubling
    @use_filter_doubling.setter
    def use_filter_doubling(self, flag):
        """ use filter doubling within a layer """
        if not isinstance(flag, bool):
            raise TypeError('use_filter_doubling should be a boolean flag')
        self._use_filter_doubling = flag


    def logits(self):
        """ return the un-normalized logits (i.e. last) layer of the network """
        return self._net[-1]

    def build(self, features):
        """ build

        Build the network using the given parameters and the features. Returns
        the final output layers. Input are the features as a tensor, typically
        from a tensorflow Dataset object.
        """
        # output some details of the net
        logger.info('Building UNet ({0:s})...'.format(self.__class__.__name__))

        input_layer = self.reshape_input(features)

        # BUILD THE NET!
        self._net = [self.down_layer(input_layer, self.filters[0], name='L0d')]

        # do the down layers
        for i, f in enumerate(self.filters[1:]):
            name = "L{0:d}d".format(i+1)
            prev_layer = self.max_pool_layer(self._net[-1])
            self._net.append( self.down_layer(prev_layer, f, name=name) )

        # now add the up layers
        for i, f in reversed(list(enumerate(self.filters[:-1]))):
            name = "L{0:d}u".format(i)
            prev_layer = self._net[-1] # layer below
            bridge = self._net[i]      # bridge information
            self._net.append( self.up_layer(prev_layer, f, bridge, name=name) )

        # make an output layer with a 1x1 convolution
        logits = self.conv_layer_1x1(self._net[-1], self.n_outputs)

        logger.info('Output layer -> shape {0:s}'.format(str(logits.shape)))

        # append this layer for completeness
        self._net.append(logits)

        logger.info('...Done')

        return logits



    def down_layer(self, input_layer, filters, name=None):
        """ down_layer

        A down layer of the UNet. These are characterised by a series of
        convolution and ReLu operations, followed by a max pool to down
        sample to the next layer. A layer here is defined as 2x
        [3x3 convolution, ReLu]

        Tensor shape is often of the format: NHWC
        """

        conv1 = self.conv_layer(input_layer, filters)
        conv2 = self.conv_layer(conv1, filters)

        logger.info('Down layer -> shape {0:s}'.format(str(conv2.shape)))

        # Dropout
        drop = tf.layers.dropout(inputs=conv2,
                                 rate=self.dropout,
                                 training=self.training)
        return drop


    def up_layer(self, input_layer, filters, bridge, name=None):
        """ up_layer

        These are characterised by a series of convolution and ReLu
        operations, followed by a transpose deconvolution to up sample to the
        next layer.

        Tensor shape is often of the format: NHWC
        """

        logger.info('Up layer -> shape {0:s} (bridge: {1:s})'
                    .format(str(input_layer.shape), self.bridge_type))

        # scale up the image
        upscale = self.conv_transpose_layer(input_layer, filters)

        # now we need to incorporate the filters using the bridge
        bridge = self.bridge(upscale, bridge)

        # do the convolutions
        conv1 = self.conv_layer(bridge, filters)
        conv2 = self.conv_layer(conv1, filters)

        # dropout
        drop = tf.layers.dropout(inputs=conv2,
                                 rate=self.dropout,
                                 training=self.training)
        return drop


    def reshape_input(self, features):
        """ Reshape the input layer from the dataset features """
        raise NotImplementedError


    def conv_layer(self, input_layer, filters):
        """ Convolution layer, conv-relu with padding """
        raise NotImplementedError


    def conv_layer_1x1(self, input_layer, filters):
        """ Return a 1x1 convolution layer """
        raise NotImplementedError


    def conv_transpose_layer(self, input_layer, filters):
        """ Transpose convolution (aka deconvolution) layer """
        raise NotImplementedError

    def max_pool_layer(self, input_layer):
        """ Max pool operation """
        raise NotImplementedError


if __name__ == "__main__":
    pass
