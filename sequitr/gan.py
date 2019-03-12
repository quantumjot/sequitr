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
# Created:  23/03/2019
#-------------------------------------------------------------------------------

import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.tools import freeze_graph

# sequitr related imports
import utils
from dataio import tifffile as t

MODELDIR = "/home/alan/documents/training_data/competition_GAN"

# switch the backend for headless server
# plt.switch_backend('agg')








# handy functions
def k_leaky_relu_alpha(features, **kwargs):
    """ set up a leaky ReLU with alpha of 0.2 """
    return tf.nn.leaky_relu(features, alpha=0.2)


def pixel_norm(x, epsilon=1e-8):
    """ pixelnorm replacement for batch normalization as used in ProGAN """
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)



k_init = tf.initializers.random_normal
b_init = tf.constant_initializer(0.)




def weighted_conv2d(inputs=None,
                    filters=None,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.leaky_relu,
                    name='conv',
                    reuse=tf.AUTO_REUSE,
                    norm=True):
    """ perform a weighted initialization of a 2d convolution """

    with tf.variable_scope(name, reuse=reuse):

        # e.g. [3,3,2,64]
        k_shape = kernel_size + [inputs.shape[-1], filters]
        k_shape_prod = float(kernel_size[0]*kernel_size[1]*filters)

        # perform weighting of the kernels byt the size of the layer [3,3,2]
        kernels = tf.get_variable('filter', k_shape, initializer=k_init)
        w_kernels = tf.scalar_mul(tf.sqrt(2.0/k_shape_prod), kernels)

        # bias
        bias = tf.get_variable('bias', [1, 1, 1, filters], initializer=b_init)

        # do the convolution and add the bias
        output = tf.nn.conv2d(inputs,
                              w_kernels,
                              [1, 1, 1, 1],
                              padding=padding.upper(),
                              data_format='NHWC') + bias

        # now the activation function
        if activation:
            output = activation(output)

        # we can also do the pixel normalization here
        if norm:
            output = pixel_norm(output)

        return output


def to_image(X, filters=2, n=None):
    """ 1x1 convolution layer to convert output to an image """
    output = weighted_conv2d(inputs=X,
                             filters=filters,
                             kernel_size=[1, 1],
                             activation=None, #tf.nn.tanh,
                             padding="same",
                             name='to_image{}'.format(n),
                             reuse=tf.AUTO_REUSE,
                             norm=False)

    return output

def from_image(X, filters=2, n=None):
    """ 1x1 convolution layer to convert image to an input """
    output = weighted_conv2d(inputs=X,
                             filters=filters,
                             kernel_size=[1, 1],
                             activation=k_leaky_relu_alpha,
                             padding="same",
                             name='from_image{}'.format(n),
                             reuse=tf.AUTO_REUSE,
                             norm=True)
    return output


def half_size(X):
    """ half the size of an image """
    new_shape = tf.shape(X)[1:3] / 2
    return tf.image.resize_nearest_neighbor(X, new_shape, align_corners=True)

def double_size(X):
    """ double the size of an image """
    new_shape = tf.shape(X)[1:3] * 2
    return tf.image.resize_nearest_neighbor(X, new_shape, align_corners=True)












def discriminator_network(x, filters):
    """ discriminator_network

    The discriminator network is essentially a standard convolutional
    neural network classifier. Build the full network in one shot.

    Returns a 'probability' of being real.

    Notes:
         - this doesn't use a 1x1 convolution layer at the start. That
           needs to be added separately since we may be feeding in to a
           lower layer when training.
    """

    # store the input layers so that we can inject data at the correct scale
    num_layers = len(filters)

    # build an input here from an image
    with tf.variable_scope("from_image", auxiliary_name_scope=True):
        x = from_image(x, filters=filters[0], n=num_layers-1)
        conv_layers = [x]

    for l, f in enumerate(filters[1:]):
        with tf.variable_scope("layer_{0:d}".format(num_layers-l-1)):
            # convolutional layers
            conv1 = weighted_conv2d(inputs=conv_layers[-1],
                                    filters=f,
                                    kernel_size=[3, 3],
                                    padding="same",
                                    activation=k_leaky_relu_alpha,
                                    name='conv1',
                                    reuse=tf.AUTO_REUSE,
                                    norm=False)

            conv2 = weighted_conv2d(inputs=conv1,
                                    filters=f,
                                    kernel_size=[3, 3],
                                    padding="same",
                                    activation=k_leaky_relu_alpha,
                                    name='conv2',
                                    reuse=tf.AUTO_REUSE,
                                    norm=False)

            # pooling Layer
            p = tf.layers.average_pooling2d(inputs=conv2,
                                            pool_size=[2, 2],
                                            strides=2,
                                            name='pool')

            # make the new output, the input of the previous layer
            conv_layers.append(p)

    # input to the final layers
    x = conv_layers[-1]

    with tf.variable_scope('output'):
        # minibatch standard deviation layer
        # compute the stdev of each feature at each spatial position across
        # the mini batch
        mean, var = tf.nn.moments(x,
                                  [0],
                                  shift=None,
                                  name='moments',
                                  keep_dims=True)

        var = tf.reduce_mean(var, keepdims=False)
        stdev = tf.sqrt(var, name='minibatch_stdev')
        minibatch_stdev = tf.ones((tf.shape(x)[0],4,4,1), tf.float32) * stdev

        # convolutional layer
        conv = weighted_conv2d(inputs=x,
                               filters=filters[-1],
                               kernel_size=[3, 3],
                               padding="same",
                               activation=k_leaky_relu_alpha,
                               name='conv',
                               reuse=tf.AUTO_REUSE,
                               norm=False)

        conv = tf.concat([conv, minibatch_stdev], axis=-1)

        pool_flat = tf.reshape(conv, [-1, 4*4*(filters[-1]+1)]) # +1 due to minibatch std
        dense = tf.layers.dense(inputs=pool_flat,
                                units=filters[-1],
                                activation=k_leaky_relu_alpha,
                                name='dense',
                                reuse=tf.AUTO_REUSE)

        # logits Layer
        logits = tf.layers.dense(inputs=dense,
                                 units=1,
                                 name='logits',
                                 reuse=tf.AUTO_REUSE)


    return conv_layers, tf.squeeze(logits)





def generator_network(z, filters, start_shape=(4,4)):
    """ The generator network takes a vector of random noise and generates
    an output imageself.

    Notes:
         - this doesn't use a 1x1 convolution layer at the end. That
           needs to be added separately since we may be extracting a
           lower layer when training.
    """

    with tf.variable_scope('latent'):

        # pop the first filter, this ensures we make the correct number of
        # convolutional layers later on...

        # calculate the number of features
        initial_shape = start_shape+(filters[0],)
        num_units = np.prod(initial_shape)

        dense = tf.layers.dense(inputs=pixel_norm(z),
                                units=num_units,
                                activation=k_leaky_relu_alpha,
                                name='dense1',
                                reuse=tf.AUTO_REUSE)

        # pixel norm and reshape to an image
        reshaped = pixel_norm(tf.reshape(dense, (-1,)+initial_shape))

        # # first convolutional layer
        conv0 = weighted_conv2d(inputs=reshaped,
                                 filters=filters[0],
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=k_leaky_relu_alpha,
                                 name='conv',
                                 reuse=tf.AUTO_REUSE,
                                 norm=True)

    # now set the value of x (the output of the previous layer)
    conv_layers = [conv0]

    # now build the layers
    for l, f in enumerate(filters[1:]):
        with tf.variable_scope('layer_{0:d}'.format(l)):

            upscale = double_size(conv_layers[-1])

            # convolutional layers
            conv1 = weighted_conv2d(inputs=upscale,
                                    filters=f,
                                    kernel_size=[3, 3],
                                    padding="same",
                                    activation=k_leaky_relu_alpha,
                                    name='conv1',
                                    reuse=tf.AUTO_REUSE,
                                    norm=True)

            conv2 = weighted_conv2d(inputs=conv1,
                                    filters=f,
                                    kernel_size=[3, 3],
                                    padding="same",
                                    activation=k_leaky_relu_alpha,
                                    name='conv2',
                                    reuse=tf.AUTO_REUSE,
                                    norm=True)

            # convolutional layer
            conv_layers.append(conv2)

    outputs = []
    with tf.variable_scope("to_image", auxiliary_name_scope=True):
        for l, conv in enumerate(conv_layers):
            output = to_image(conv, filters=2, n=l)
            outputs.append(output)

    return outputs, output






























def tr_input_fn(record_file, num_epochs=None, batch_size=1, params={}):
    """ Take the input data path and return an iterator to the dataset """

    tr_data = (tf.data.TFRecordDataset([record_file])
                .map(lambda x:tr_input_parser(x))
                .cache()
                .shuffle(buffer_size=1536)
                .map(lambda x:tr_augment(x,params), num_parallel_calls=4)
                .batch(batch_size)
                .repeat(num_epochs)
                .prefetch(batch_size))

    # create TensorFlow Iterator object
    return tr_data.make_initializable_iterator()

def tr_input_parser(serialized_example):
    """ Parse input images """

    # set up the fixed length features to load
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/width': tf.FixedLenFeature([], tf.int64),
               'train/height': tf.FixedLenFeature([], tf.int64),
               'train/channels': tf.FixedLenFeature([], tf.int64)}

    features = tf.parse_single_example(serialized_example, features=feature)

    # convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.uint8)

    # get the size of the images
    height = tf.cast(features['train/height'], tf.int32)
    width = tf.cast(features['train/width'], tf.int32)
    channels = tf.cast(features['train/channels'], tf.int32)
    image_shape = tf.stack([height, width, channels])

    # reshape image data into the original shape
    img = tf.cast(tf.reshape(image, image_shape), tf.float32)

    # normalize the image
    mean, var = tf.nn.moments(img, axes=[0,1], keep_dims=True)
    img = tf.nn.batch_normalization(img,
                                    mean,
                                    var,
                                    None, None, 1e-8,
                                    name='dataset_image_normalization')

    # need to normalise the input images to -1 to 1
    # img = (img/127.5)-1.
    return img

def tr_augment(features, params):
    """ Reshape, randomly crop and flip """

    # random crop
    img = tf.image.random_crop(features, (512,512,2))

    # now do some random flips
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img






# def init_new_variables(session, variables=None):
#     """ Initialize only new variables """
#     all_vars = tf.global_variables() + tf.local_variables()
#     get_var = {v.op.name: v for v in all_vars}
#     uninit_var_names = session.run(tf.report_uninitialized_variables(variables))
#     uninit_vars = [get_var[v] for v in uninit_var_names]
#     init_op = tf.variables_initializer(uninit_vars)
#     session.run(init_op)





class GAN2DConfiguration(utils.NetConfiguration):
    """ GAN2DConfiguration

    A default configuration for a GAN2D class. Several of these parameters can
    be overwritten when specifying a job from the server.

    num_expansion is the number of expansions from a 4x4 starting image.
    """

    def __init__(self, params=None):
        utils.NetConfiguration.__init__(self)
        self.name = 'GAN_competition'
        # self.dropout = 0.4
        self.batch_size = 32
        self.repeat_batch = 4
        self.num_outputs = 2
        self.num_levels = 7
        self.num_epochs_per_level = 1
        self.start_size = (4,4)
        self.learning_rate = 1e-3
        self.warm_start = False
        self.path = ''
        self.training_data = 'train_GAN.tfrecord'


class GenerativeAdverserialNetwork(object):
    """ Generative Adverserial Network

    This is a GAN (based on DCGAN and ProGAN, references below) to synthesize
    microscopy images.


    Before training:
        The network needs training data. Before running, use the
        create_GAN_tfrecord function to create a serialized tfrecord of the
        training data. This is much more efficient for dataio. See the
        documentation for that function for how to do this.

        # must set up a tfrecord file of training data
        create_GAN_tfrecord(src, filename, n_samples=256)


    Args:
        params:             a dictionary of parameters from the config
        mode:               training mode (DEPRECATED)
        discriminator_fn:   a function that builds a discriminator network
        generator_fn:       a function that builds a generator network

    Use:
        # set up a GAN configuration
        config = GAN2DConfiguration()
        config.batch_size = 32
        config.training_data = "train_GAN.tfrecord"

        # set up the GAN
        mode = tf.estimator.ModeKeys.TRAIN
        gan = GenerativeAdverserialNetwork(config.to_params(), mode)
        gan.output_dir = "/mnt/lowe-sn00/JobServer/output/GAN/"

        # train it
        gan.train()


    Notes:
        Generative Adversarial Networks
        Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu,
        David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
        https://arxiv.org/abs/1406.2661

        Progressive Growing of GANs for Improved Quality, Stability,
        and Variation
        Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
        https://arxiv.org/abs/1710.10196

    """
    def __init__(self,
                 params,
                 mode=None,
                 discriminator_fn=discriminator_network,
                 generator_fn=generator_network):

        self.__discriminator_fn = discriminator_fn
        self.__generator_fn = generator_fn
        self.saver = None

        # NOTE(arl): this is a hang-over from the Estimator config interface
        self.__num_channels = params.get('num_outputs', 2)
        self.batch_size = params.get('batch_size', 32)
        self.repeat_batch = params.get('repeat_batch', 1)
        self.num_levels = params.get('num_levels', 3)
        self.num_epochs_per_level = params.get('num_epochs_per_level', 10)
        self.start_size = params.get('start_size', (4,4))
        self.training_data_filename = params.get('training_data', None)
        self.learning_rate = params.get('learning_rate', 1e-3)

        # get the number of entries in the tf_record_file
        n = len([x for x in tf.python_io.tf_record_iterator(self.training_data_filename)])
        self.num_batches_per_epoch = int(n/self.batch_size)

        # set up the dataset
        self.dataset = tr_input_fn(self.training_data_filename, None, self.batch_size)

        # somewhere to store the networks
        self.networks = []

        # store the mode and params
        self.mode = mode
        self.__params = params

        # store the expansion iteration
        self.__level = 0

        # set up the sequence of filters
        self.filters = utils.filter_doubling(start_filters=8,
                                             num_layers=self.num_levels,
                                             max_filters=512,
                                             reverse=True)


    @property
    def num_channels(self):
        return self.__num_channels

    @property
    def current_level(self):
        """ return the expansion iteration """
        return self.__level

    @property
    def num_iterations_this_level(self):
        """ return the number of iteration for this expansion """
        return self.num_epochs_per_level*self.num_batches_per_epoch
        # return 48 * (20*2**self.expansion_iteration)

    # @property
    # def layer_shapes(self):
    #     """ return the final output size of the image """
    #     return [(self.start_size[0]*(2**l), self.start_size[1]*(2**l)) for l in range(self.num_levels+1)]

    def get_size(self, level):
        return tuple([s*(2**level) for s in self.start_size])

    @property
    def current_size(self):
        """ return the current output size """
        return self.get_size(self.current_level)


    def expand(self):
        """ Expand the network output size """
        self.__level+=1
        assert(self.__level <= self.num_levels)


    # @utils.network_device_placement("/gpu:1")
    def generator(self, Z, filters, **kwargs):
        """ proxy for the generator network. """

        with tf.variable_scope('generator'):
            return self.__generator_fn(Z, filters, **kwargs)


    # @utils.network_device_placement("/gpu:0")
    def discriminator(self, X, filters, **kwargs):
        """ proxy for the discriminator network. """
        with tf.variable_scope('discriminator'):
            return self.__discriminator_fn(X, filters, **kwargs)


    def get_training_variables(self, current_layer):
        """ return the training variables for the whole GAN """
        # # set up the optimizers to update their respective vars
        d_vars = self.discriminator_training_variables(current_layer)
        g_vars = self.generator_training_variables(current_layer)
        d_input_vars = tf.trainable_variables(scope='GAN/discriminator/from_image/from_image{0:d}'.format(current_layer))
        g_output_vars = tf.trainable_variables(scope='GAN/generator/to_image/to_image{0:d}'.format(current_layer))
        # if current_layer > 0:
        #     g_output_vars += tf.trainable_variables(scope='GAN/generator/output/to_image{0:d}'.format(current_layer-1))
        d_vars += d_input_vars
        g_vars += g_output_vars
        print current_layer, " --> D_VARS:", d_vars
        print current_layer, " --> G_VARS:", g_vars
        return d_vars, g_vars


    def generator_training_variables(self, current_layer):
        """ return the training variables for the generator network """
        inputs = tf.trainable_variables(scope='GAN/generator/latent')
        vars = []
        for layer in range(current_layer):
            vars += tf.trainable_variables(scope='GAN/generator/layer_{0:d}'.format(layer))
        return vars + inputs

    def discriminator_training_variables(self, current_layer):
        """ return the training variables for the discriminator network """
        outputs = tf.trainable_variables(scope='GAN/discriminator/output')
        vars = []
        for layer in range(current_layer):
            vars += tf.trainable_variables(scope='GAN/discriminator/layer_{0:d}'.format(layer))
        return vars + outputs




    def build(self, model):
        """ build the network(s) and placeholders """
        self._build_placeholders()
        self._build_networks()

        self.initialized = True


    def _build_placeholders(self):
        """ set up the placeholders """
        # feed placeholders
        self.X = tf.placeholder(tf.float32, shape=(None, None, None, self.num_channels), name='X')
        self.Z = tf.placeholder(tf.float32, shape=(None, 1, 1, 512), name='Z')
        self.alpha = tf.placeholder(tf.float32, shape=(), name='alpha')
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')




    def _build_networks(self, levels_to_build):
        """ build all of the networks in one go """

        # set up the optimizers
        d_opt, g_opt = self._build_optimizers()

        # now build all of the networks
        for i in range(self.num_levels):
            Gz, d_loss, g_loss = self._build_network(self.X, self.Z, self.alpha)

            d_vars, g_vars = self.get_training_variables(i)
            d_solver = d_opt.minimize(d_loss, var_list=d_vars)
            g_solver = g_opt.minimize(g_loss, var_list=g_vars,
                                      global_step=self.global_step)

            # self._build_summaries(Gz, d_loss, g_loss)

            self.networks.append((Gz, d_loss, g_loss, d_solver, g_solver))

            self.expand()







    def _build_network(self, X, Z, alpha):
        """ build the whole network """

        num_layers = self.current_level
        filters = self.filters[:(num_layers+1)]
        d_filters_crop = filters[::-1]  # reversed
        g_filters_crop = filters

        print num_layers, d_filters_crop, g_filters_crop

        print "Building new graph with {0:d} layers...".format(num_layers)
        with tf.variable_scope('GAN', reuse=tf.AUTO_REUSE):

            # generator discriminator pair
            g_layers, Gz_raw = self.generator(Z, g_filters_crop)

            # resize the real input
            X_resized = tf.image.resize_images(X,
                                               self.current_size,
                                               align_corners=True)

            # make a mixed output if we have more than one layer...
            if num_layers > 0:

                prev_Gz = double_size(g_layers[-2])
                Gz = tf.add(alpha*Gz_raw, (1.-alpha)*prev_Gz)

                # do the same for the real data
                prev_X = double_size(half_size(X_resized))
                X_resized = tf.add(alpha*X_resized, (1.-alpha)*prev_X)

            else:
                Gz = Gz_raw

            # discriminator for arbitrary inputs
            d_layers, Dz = self.discriminator(Gz, d_filters_crop)
            _, Dx = self.discriminator(X_resized, d_filters_crop)

            print X_resized.shape, Gz.shape

            # sanity check, make sure the same number of layers in each network
            assert(len(g_layers) == len(d_layers))

            # operations to mix the input and generated
            r = tf.random_uniform(shape=[self.batch_size, 1, 1, 1],
                                  minval=0.0,
                                  maxval=1.0,
                                  dtype=tf.float32)
            mix = tf.add(r*X_resized, (1-r)*Gz)
            _, Dmix = self.discriminator(mix, d_filters_crop)


        # now make the Wasserstein Loss Function
        with tf.variable_scope('WGAN-GP_loss'):
            loss_lambda = 10.

            grad = tf.gradients(Dmix, [mix])[0]
            grad_normed = tf.sqrt(tf.reduce_sum(tf.square(grad),[1,2,3]))
            # grad_penalty = tf.square((grad_normed-1.0))
            lipschitz_penalty = tf.square(tf.maximum((grad_normed-1.0), 0))
            scaled_penalty = loss_lambda*lipschitz_penalty
            eps_penalty = 0.001 * tf.square(Dx)

            g_loss = tf.reduce_mean(-Dz)
            d_loss = tf.reduce_mean(-Dx + Dz + scaled_penalty + eps_penalty)

        # hand back the un-mixed Gz
        return Gz_raw, d_loss, g_loss



    def _build_optimizers(self, level=0):
        """ build optimizers for the discriminator and generator """

        with tf.variable_scope('optimizers'):
            d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.,
                                           beta2=0.99,
                                           name='d_solver_{0:d}'.format(level))

            g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.,
                                           beta2=0.99,
                                           name='g_solver_{0:d}'.format(level))


        return d_opt, g_opt

        # l = self.current_level
        #
        # # set up AdamOptimizer for the discriminator and generator
        # d_vars, g_vars = self.get_training_variables(l)
        # d_solver = d_opt.minimize(d_loss, var_list=d_vars)
        # g_solver = g_opt.minimize(g_loss, var_list=g_vars, global_step=global_step)
        #



    def _build_summaries(self, d_loss, g_loss, Gz, level=0):
        # set up an image summary of the output (use green, magenta)
        layer_id = "{0}x{1}".format(*self.get_size(level))
        tf.summary.image("Gz_"+layer_id, tf.concat([Gz[...,1:2],
                                                    Gz[...,0:1],
                                                    Gz[...,1:2]], axis=-1),
                                                family="generator")
        tf.summary.scalar("generator_"+layer_id, g_loss, family="loss")
        tf.summary.scalar("discriminator_"+layer_id, d_loss, family="loss")



    def build_latent(self):
        # build the latent vector generator
        with tf.variable_scope('latent_random_normal'):
            Z_ = tf.random.normal((self.batch_size, 1, 1, 512),
                                   mean=0.,
                                   stddev=1.,
                                   dtype=tf.float32,
                                   seed=None,
                                   name='Z')

        return Z_

    def checkpoint(self, sess):
        pass

    # @utils.as_tf_session(config=tf.ConfigProto(log_device_placement=True))
    @utils.as_tf_session()
    def train(self, session=None):
        """ Do the training in a session. """

        if not self.initialized:
            raise Exception("Networks have not been initialized. Please run .build()")

        # get the dataset iterator, and latent vector
        data = self.dataset.get_next()
        Z_ = self.build_latent()

        # set up a file writer with most of the graph...
        utils.check_and_makedir(self.output_dir)
        writer = tf.summary.FileWriter(self.output_dir)
        saver = tf.train.Saver()

        # save out the graph and merge summaries
        writer.add_graph(session.graph, global_step=0)
        tf.summary.scalar("alpha", self.alpha, family="alpha")
        summaries = tf.summary.merge_all()

        # get the value of the global step as an integer
        def gs(): return int(session.run(self.global_step))

        # INITIALIZE!!
        session.run(self.dataset.initializer)
        session.run(tf.global_variables_initializer())

        for n, network in enumerate(self.networks):

            # unzip it
            Gz, d_loss, g_loss, d_solver, g_solver = network
            self._build_summaries(d_loss, g_loss, Gz, n)
            summaries = tf.summary.merge_all()

            for phase in ('fade', 'stabilisation'):
                for step in range(self.num_iterations_this_level):

                    if phase == 'fade':
                        fade = float(step+1)/(self.num_iterations_this_level)
                    else:
                        fade = 1.0

                    # get a random sample and a real image
                    z, x = session.run([Z_, data])

                    # set up the feed dictionary
                    feed = {self.X: x, self.Z: z, self.alpha: fade}

                    # evaluate  the losses
                    for r in range(self.repeat_batch):
                        session.run(d_solver, feed_dict=feed)
                        session.run(g_solver, feed_dict=feed)

                    print gs(), phase, self.get_size(n), fade

                    if gs() % (100*self.repeat_batch) == 0:
                        print gs(), "Updating summaries..."
                        # get the summaries...
                        summary = session.run(summaries, feed_dict=feed)
                        writer.add_summary(summary, global_step=gs())
                        writer.flush()

            # save out the model
            print gs(), "Checkpoint..."
            model_fn = "model_{0:s}.ckpt".format(str(self.current_size).replace(', ','x'))
            saver.save(session, os.path.join(self.output_dir, model_fn))

        writer.close()
        saver.close()

    @utils.as_tf_session()
    def predict(self, session=None, Z):
        """ given some latent vector Z, generate some images. """
        pass







def create_GAN_dataset(filename, src_pth, dirs, channels):
    # create a GAN dataset
    # src_pth = "/mnt/lowe-sn00/Data/Alan/Anna_to_process/"
    # src_pth = "/media/arl/DataII/Data/competition/RNN/"

    dirs = ['2017_03_31/pos7',
            '2017_03_31/pos9',
            '2017_03_31/pos11',
            '2017_03_31/pos13',
            '2017_03_31/pos15',
            '2017_03_31/pos17']
    channels = ["gfp", "rfp"]

    src = []
    for d in dirs:
        channel_files = []
        for c in channels:
            channel = os.listdir(os.path.join(src_pth, d, c+"/"))
            channel = [cf for cf in channel if cf.startswith((c,c.upper()))]
            channel_files.append(os.path.join(src_pth, d, c, channel[0]))
        src.append([s for s in channel_files])

    print src

    create_GAN_tfrecord(src, filename, n_samples=256)

def create_GAN_tfrecord(src,
                        filename,
                        n_samples=128):

    """ create_GAN_tfrecord

    Create a TFRecord file for classifier examples. These are generally of the
    format:

        image: a multidimensional image WxHxC
        label: a single integer label

    """

    # _int64 is used for numeric values
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # _bytes is used for string/char values
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # _floats is used for float values
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    import random
    from scipy.ndimage.filters import median_filter

    # set up the writer
    writer = tf.python_io.TFRecordWriter(filename)

    # found_labels = find_folders(pth)
    for gfp_fn, rfp_fn in src:

        print gfp_fn, rfp_fn

        img_gfp = t.imread(gfp_fn).astype('uint8')
        img_rfp = t.imread(rfp_fn).astype('uint8')

        assert(img_gfp.shape == img_rfp.shape)

        # stack the data along a new axis (final, e.g. w,h,c)
        img_stack = np.stack((img_gfp,img_rfp), axis=-1)
        img_shape = img_stack.shape

        # choose some random indices
        idx = range(img_shape[0])
        random.shuffle(idx)
        idx = idx[0:n_samples]

        while idx:
            n = idx.pop(0)

            img_data = img_stack[n,...]

            # do a quick cleanup
            threshold = 5.
            for chnl in xrange(img_data.shape[-1]):
                filtered_image = img_data[...,chnl].copy()
                med_filtered_image = median_filter(filtered_image, 2)
                diff_image = np.abs(img_data[...,chnl] - med_filtered_image)
                differences = diff_image>threshold
                filtered_image[differences] = med_filtered_image[differences]
                img_data[...,chnl] = filtered_image.astype('uint8')

            print n, img_shape, img_data.shape

            feature = {'train/image': _bytes_feature(img_data.tostring()),
                       'train/width': _int64_feature(img_shape[2]),
                       'train/height': _int64_feature(img_shape[1]),
                       'train/channels': _int64_feature(img_shape[3])}

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # write out the serialized features
            writer.write(example.SerializeToString())

    # close up shop
    writer.close()











if __name__ == "__main__":

    outdir = "/mnt/lowe-sn00/JobServer/output/"

    import argparse

    p = argparse.ArgumentParser(description='Sequitr: Progressive GAN')
    p.add_argument('--outdir', default=outdir,
                    help='Path to job directory')
    p.add_argument('--num_epochs', type=int, default=100,
                    help='Specify the number of epochs per expansion')
    p.add_argument('--num_levels', type=int, default=8,
                    help='Specify the number of expansions from (4,4) start')
    p.add_argument('--batch_size', type=int, default=32,
                    help='Specify the batch size')
    # p.add_argument('--use_GP', action='store_true',
    #                 help='Use gradient penalty rather than Lipschitz.')

    args = p.parse_args()
    print args


    filename = os.path.join(MODELDIR, "train_GAN.tfrecord")

    # # get the configuration
    config = GAN2DConfiguration()
    config.num_levels = args.num_levels
    config.num_epochs_per_level = args.num_epochs
    config.batch_size = args.batch_size
    config.training_data = filename

    def get_next_run_number(folder):
        runs = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f)) if f.startswith('GAN')]
        if not runs:
            return 0
        else:
            return max([int(r.lstrip('GAN')) for r in runs])+1


    output_dir = os.path.join(outdir,"GAN{0:d}".format(get_next_run_number(outdir)))
    print output_dir


    # set up the GAN
    mode = tf.estimator.ModeKeys.TRAIN
    gan = GenerativeAdverserialNetwork(config.to_params(), mode)
    gan.output_dir = output_dir

    gan.build()
    gan.train()
