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
plt.switch_backend('agg')


# convenience definitions to prevent spelling errors when naming layers...
D_NET = 'discriminator'
G_NET = 'generator'






# handy functions
def k_leaky_relu_alpha(features, **kwargs):
    """ set up a leaky ReLU with alpha of 0.2 """
    return tf.nn.leaky_relu(features, alpha=0.2)


def pixel_norm(x, epsilon=1e-8):
    """ pixelnorm replacement for batch normalization as used in ProGAN """
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

k_init = tf.contrib.layers.xavier_initializer(uniform=False)



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
        self.num_outputs = 2
        self.num_expansions = 7
        self.num_epochs_per_expansion = 500
        self.start_size = (4,4)
        self.learning_rate = 1e-3
        self.balance_loss = False
        self.warm_start = False
        self.path = ''
        self.training_data = 'train_GAN.tfrecord'



def discriminator_network(features,
                          input_shape,
                          num_filters=64,
                          num_layers=0,
                          max_num_filters=512,
                          mode=tf.estimator.ModeKeys.TRAIN):
    """ The discriminator network is essentially a standard convolutional
    neural network classifier.

    Returns a 'probability' of being real.
    """

    # use a 1x1 convolution to maker sure we always have the same number of
    # inputs, that is larger than 2...

    # set up the filter doubling
    # layers = [(l, int(min((2**(9-l), 512)))) for l in range(9)]
    layers = [(l, num_filters) for l in range(9)]
    layers.reverse()

    if num_layers == 0:
        layers_to_build = []
    else:
        layers_to_build = layers[-num_layers:]

    # in filters is the number in the next layer to be added (confusing!)
    filters_in = layers[-(num_layers+1)][1]

    x = tf.layers.conv2d(inputs=features,
                         filters=filters_in,
                         kernel_size=[1, 1],
                         padding="same",
                         activation=tf.nn.leaky_relu,
                         reuse=tf.AUTO_REUSE,
                         name=D_NET+'/conv1x1')
                         # name=D_NET+'/conv1x1_{0:d}'.format(num_layers))

    layer_shape = input_shape
    f = layers[-1][1]

    for layer_id, f in layers_to_build:

        # convolutional layer
        c = tf.layers.conv2d(inputs=x,
                             filters=f,
                             kernel_size=[3, 3],
                             padding="same",
                             activation=tf.nn.leaky_relu,
                             name=D_NET+'/conva{0:d}'.format(layer_id),
                             reuse=tf.AUTO_REUSE)

        # convolutional layer
        c2 = tf.layers.conv2d(inputs=c,
                             filters=f,
                             kernel_size=[3, 3],
                             padding="same",
                             activation=tf.nn.leaky_relu,
                             name=D_NET+'/convb{0:d}'.format(layer_id),
                             reuse=tf.AUTO_REUSE)

        # pooling Layer
        p = tf.layers.average_pooling2d(inputs=c2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name=D_NET+'/pool{0:d}'.format(layer_id))

        # update some of the parameters
        layer_shape = tuple([lsz/2 for lsz in layer_shape])
        x = p

    # flatten the layer
    assert(layer_shape == (4,4))
    # assert(f == layers[-1][1])

    # convolutional layer
    conv = tf.layers.conv2d(inputs=x,
                            filters=f,
                            kernel_size=[3, 3],
                            padding="same",
                            activation=tf.nn.leaky_relu,
                            name=D_NET+'/conv_final',
                            reuse=tf.AUTO_REUSE)

    # pool_flat = tf.reshape(x, [-1, layer_shape[0]*layer_shape[1]*f])
    pool_flat = tf.reshape(conv, [-1, 4*4*f])
    dense = tf.layers.dense(inputs=pool_flat,
                            units=1024,
                            activation=tf.nn.leaky_relu,
                            name=D_NET+'/dense',
                            reuse=tf.AUTO_REUSE)

    # logits Layer
    logits = tf.layers.dense(inputs=dense,
                             units=1,
                             name=D_NET+'/logits',
                             reuse=tf.AUTO_REUSE)

    return tf.squeeze(logits)





def generator_network(batch_size,
                      num_filters=64,
                      num_layers=0,
                      input_shape=(4,4,2),
                      mode=tf.estimator.ModeKeys.TRAIN):
    """ The generator network takes a vector of random noise and generates
    an output image """

    # set the training mode for batch norm
    is_training = mode==tf.estimator.ModeKeys.TRAIN

    input_layer = tf.random.normal((batch_size,128),
                                   mean=0.,
                                   stddev=1.,
                                   dtype=tf.float32,
                                   seed=None,
                                   name=G_NET+'/input_layer')

    # pixel norm the input layer
    input_layer = pixel_norm(input_layer)

    dense = tf.layers.dense(inputs=input_layer,
                            units=np.prod(input_shape),
                            activation=k_leaky_relu_alpha,
                            name=G_NET+'/dense',
                            reuse=tf.AUTO_REUSE)

    # pixel norm and reshape to an image
    dense = pixel_norm(tf.reshape(dense, (batch_size,)+input_shape))

    # first convolutional layer
    convd = tf.layers.conv2d(inputs=dense,
                             filters=num_filters,
                             kernel_size=[3, 3],
                             padding="same",
                             activation=k_leaky_relu_alpha,
                             name=G_NET+'/conv_dense',
                             reuse=tf.AUTO_REUSE)

    # final pixel norm before upsclaing begins
    convd = pixel_norm(convd)


    # now set the value of x (the output of the previous layer)
    # x = convd
    layers = [convd]


    for l in range(num_layers):

        # use filter doubling to set the number of filters
        # f = min(start_filters*(2**(l+1)), max_num_filters)

        # upscale the image
        new_size = [sz*(2**(l+1)) for sz in input_shape[0:2]]
        t = tf.image.resize_nearest_neighbor(layers[-1],
                                             new_size,
                                             align_corners=True,
                                             name=G_NET+'/upscale{0:d}'.format(l))

        # convolutional layer
        c1 = tf.layers.conv2d(inputs=t,
                              filters=num_filters,
                              kernel_size=[3, 3],
                              padding="same",
                              activation=k_leaky_relu_alpha,
                              name=G_NET+'/conva{0:d}'.format(l),
                              reuse=tf.AUTO_REUSE)

        bn1 = pixel_norm(c1)

        # convolutional layer
        c2 = tf.layers.conv2d(inputs=bn1,
                              filters=num_filters,
                              kernel_size=[3, 3],
                              padding="same",
                              activation=k_leaky_relu_alpha,
                              name=G_NET+'/convb{0:d}'.format(l),
                              reuse=tf.AUTO_REUSE)

        # now reset the value of x
        bn2 = pixel_norm(c2)
        layers.append(bn2)

    # return the layers
    return layers










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
    # iterator = tr_data.make_one_shot_iterator()
    # return iterator.get_next()
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
                                    None, None, 1e-99,
                                    name='dataset_image_normalization')

    # # need to normalise the input images to -1 to 1
    # img = (img/128.)-1.

    return img

def tr_augment(features, params):
    """ Reshape, randomly crop and flip """

    # random crop
    img = tf.image.random_crop(features, (512,512,2))

    # now do some random flips
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img




# def random_flip_labels(labels, num_random_flips=2):
#     """ randomly flip training labels """
#     new_labels = labels.copy()
#     idx = np.random.choice(labels.shape[0],
#                            num_random_flips,
#                            replace=False)
#     new_labels[idx] = 1-labels[idx]
#     return new_labels
#
# def soft_labels(labels, max_val=0.1):
#     """ make the labels soft """
#     soft_labels = labels.copy()
#     r = np.random.random(labels.shape[0])
#     soft_labels = np.abs(soft_labels - max_val*r)
#     return np.array(soft_labels)


def init_new_variables(session, variables=None):
    """ Initialize only new variables """
    all_vars = tf.global_variables() + tf.local_variables()
    get_var = {v.op.name: v for v in all_vars}
    uninit_var_names = session.run(tf.report_uninitialized_variables(variables))
    uninit_vars = [get_var[v] for v in uninit_var_names]
    init_op = tf.variables_initializer(uninit_vars)
    session.run(init_op)





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
        mode:               training mode
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
                 mode,
                 discriminator_fn=discriminator_network,
                 generator_fn=generator_network):

        self.__discriminator_fn = discriminator_fn
        self.__generator_fn = generator_fn
        self.saver = None

        # NOTE(arl): this is a hang-over from the Estimator config interface
        self.__num_channels = params.get('num_outputs', 2)
        self.batch_size = params.get('batch_size', 32)
        self.num_expansions = params.get('num_expansions', 3)
        self.num_epochs_per_expansion = params.get('num_epochs_per_expansion', 10)
        self.balance_loss = params.get('balance_loss', False)
        self.start_size = params.get('start_size', (4,4))
        self.training_data_fn = params.get('training_data', None)
        self.learning_rate = params.get('learning_rate', 1e-3)

        # get the number of entries in the tf_record_file
        n = len([x for x in tf.python_io.tf_record_iterator(self.training_data_fn)])
        self.num_batches_per_epoch = int(n/self.batch_size)

        # store the mode and params
        self.mode = mode
        self.__params = params

        # store the expansion iteration
        self.__expansion_iter = 0

        # space to store the initialized vars
        # self.initialized_vars = []


    @property
    def num_channels(self):
        return self.__num_channels

    @property
    def expansion_iteration(self):
        """ return the expansion iteration """
        return self.__expansion_iter

    @property
    def num_iterations_this_expansion(self):
        """ return the number of iteration for this expansion """
        return self.num_epochs_per_expansion*self.num_batches_per_epoch
        # return 48 * (20*2**self.expansion_iteration)

    @property
    def final_size(self):
        """ return the final output size of the image """
        return tuple([s*(2**num_expansions) for s in self.start_size])

    @property
    def current_size(self):
        """ return the current output size """
        return tuple([s*(2**self.expansion_iteration) for s in self.start_size])


    def expand(self):
        """ Expand the network output size """
        self.__expansion_iter+=1
        assert(self.__expansion_iter <= self.num_expansions)


    @utils.network_device_placement("/gpu:1")
    def generator(self, alpha, batch_size, **kwargs):
        """ proxy for the generator network. This function also fades in
        layers to prevent the 'shock' of adding new layers to the network.

        NOTE(arl): could move the fade to the generator_fn definition.
        """

        # get the generator output before 1x1 convolution layer
        gen_layers = self.__generator_fn(batch_size, **kwargs)

        # final 1x1 convolution to get the output image
        out = tf.layers.conv2d(inputs=gen_layers[-1],
                               filters=self.num_channels,
                               kernel_size=[1, 1],
                               kernel_initializer=k_init,
                               activation=tf.nn.tanh,
                               padding="same",
                               name=G_NET+'/conv1x1',
                               reuse=tf.AUTO_REUSE)

        if kwargs['num_layers']>0:
            # this is a 1x1 convolution (using the same kernels), but with the
            # fading out pre-layer
            out_pre = tf.layers.conv2d(inputs=gen_layers[-2],
                                       filters=self.num_channels,
                                       kernel_size=[1, 1],
                                       kernel_initializer=k_init,
                                       activation=tf.nn.tanh,
                                       padding="same",
                                       name=G_NET+'/conv1x1',
                                       reuse=tf.AUTO_REUSE)

            out_pre = tf.image.resize_nearest_neighbor(out_pre,
                                                       self.current_size,
                                                       align_corners=True,
                                                       name=G_NET+'/upscale_pre')

            # output is the sum of the weighted fading in and out layers
            weighted_out = tf.add(alpha*out, (1.0-alpha)*out_pre)
        else:
            weighted_out = out
        return weighted_out

    @utils.network_device_placement("/gpu:0")
    def discriminator(self, alpha, X, sz, **kwargs):
        return self.__discriminator_fn(X, sz, **kwargs)

    @property
    def combined_training_variables(self):
        """ return the training variables for the whole GAN """
        return self.generator_training_variables + self.discriminator_training_variables

    @property
    def generator_training_variables(self):
        """ return the training variables for the generator network """
        return [v for v in tf.trainable_variables() if v.name.startswith(G_NET)]

    @property
    def discriminator_training_variables(self):
        """ return the training variables for the discriminator network """
        return [v for v in tf.trainable_variables() if v.name.startswith(D_NET)]

    # @property
    # def uninitialized_variables(self):
    #     pass




    def build_network(self, alpha, X):
        """ build the whole network """

        print "Building new graph..."

        # the number of layers is the expansion iteration
        num_layers = self.expansion_iteration
        sz = self.current_size
        half_batch = int(self.batch_size/2)

        # resize input data to the correct shape
        X = tf.image.resize_images(X, self.current_size, align_corners=True)
        X_ = self.generator(alpha, self.batch_size, num_layers=num_layers)

        # now slice these/concat these for the correct inputs
        # X = tf.concat([X_rsz[0:half_batch,...], X_[0:half_batch,...]], axis=0)
        real_logits = self.discriminator(alpha, X, sz, num_layers=num_layers)
        fake_logits = self.discriminator(alpha, X_, sz, num_layers=num_layers)

        # set the loss functions
        d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_logits), real_logits)
        d_loss = 0.5 * (d_loss_fake + d_loss_real)

        g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits), fake_logits)

        return X_, d_loss, g_loss



    def checkpoint(self, sess):
        pass

    @utils.as_tf_session()
    def train(self, session=None):
        """ Do the training in a session.
        NOTE(arl): the Estimator interface currently makes this difficult. """

        # get a dataset to train with
        filename = self.training_data_fn

        # set the first run flag
        first_run = True

        # store the loss curves
        g_loss_plot = []
        d_loss_plot = []
        alpha_plot = []
        scale_plot = []

        # get a dataset iterator
        dataset_iterator = tr_input_fn(filename, None, self.batch_size)

        # make some optimizers
        d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                       beta1=0.5,
                                       name='d_solver')

        g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                       beta1=0.5,
                                       name='g_solver')

        # global iteration
        global_step = 0

        # perform a number of expansions
        while self.expansion_iteration <= self.num_expansions:

            # inputs, X - real data, X_ - generated data
            X_size = (None, 512, 512, self.num_channels)
            X = tf.placeholder(tf.float32, shape=X_size, name='X')
            alpha = tf.placeholder(tf.float32, shape=(), name='alpha')

            # set up the discriminator and generator networks
            predict, d_loss, g_loss = self.build_network(alpha, X)

            # set up the optimizers to update their respective vars
            d_solver = d_opt.minimize(d_loss, var_list=self.discriminator_training_variables)
            g_solver = g_opt.minimize(g_loss, var_list=self.generator_training_variables)

            # initialize the variables
            if first_run:
                session.run(tf.global_variables_initializer())
                session.run(dataset_iterator.initializer)
                self.saver = tf.train.Saver()
                first_run = False
            else:
                # on subsequent iterations, we only need to initialize
                # new layers that have been added to the network
                uninit = init_new_variables(session)

            # set up the real data iterator
            data = dataset_iterator.get_next()

            # iterate over the data
            for step in range(self.num_iterations_this_expansion):

                # this determines the fading in of new layers...
                a = min(1., float(step)/(0.5*self.num_iterations_this_expansion-1.))

                # get some real images from the dataset
                X_real = session.run(data)

                # do the training updates...
                f_d = {alpha: a, X: X_real}
                _, D_loss_curr = session.run([d_solver, d_loss], feed_dict=f_d)
                _, G_loss_curr = session.run([g_solver, g_loss], feed_dict=f_d)

                # store the loss for visualisation
                g_loss_plot.append(G_loss_curr)
                d_loss_plot.append(D_loss_curr)
                alpha_plot.append(a)
                scale_plot.append(self.expansion_iteration)

                # write out some stats to stdout
                if global_step % 10 == 0:
                    print "Alpha: {0:2.2f},".format(a),
                    print "Exp: {0:d},".format(self.expansion_iteration),
                    print "Sz: {0:s},".format(str(self.current_size)),
                    print "Step: {0:d}/".format(global_step),
                    print "{0:d},".format(self.num_iterations_this_expansion*self.num_expansions),
                    print "D_loss: {0:2.5f},".format(D_loss_curr),
                    print "G_loss: {0:2.5f},".format(G_loss_curr),
                    print "Model size: {0:d}".format(len(self.combined_training_variables))


                # save out some images
                if global_step % 100 == 0:
                    # get some images from the generator
                    X_gen = session.run(predict, feed_dict={alpha: a})

                    size_fn = str(self.current_size).replace(', ','x')
                    fn = "out_{0:d}_{1:s}.tif".format(global_step, size_fn)
                    snap_fn = os.path.join(self.output_dir, "snaps", fn)
                    loss_fn = os.path.join(self.output_dir, "snaps", "loss.png")

                    # make an image to save, but change from (256,256,2) to
                    # (2,256,256) for the tiff format...
                    im_to_save = np.array((1.+X_gen[0,...])*128, dtype='uint8')
                    t.imsave(snap_fn, np.rollaxis(im_to_save, -1, 0))


                    # plot the loss also
                    plt.figure()
                    plt.plot(g_loss_plot, 'r-', label='generator loss')
                    plt.plot(d_loss_plot, 'b-', label='discriminator loss')
                    plt.plot(alpha_plot, 'k:', label='alpha')
                    plt.plot(scale_plot, 'k-', label='output size')
                    plt.yscale('log')
                    plt.ylim([0.001, 100.])
                    plt.xlabel('Iterations')
                    plt.ylabel('Loss')
                    plt.title('Iterations: {0:d}'.format(global_step))
                    plt.legend()
                    plt.savefig(loss_fn, dpi=144)
                    plt.close()

                # increment the counter
                global_step+=1


            # save the model
            self.saver.save(session, os.path.join(self.output_dir, "model"), global_step=global_step)

            # expand the network
            self.expand()















if __name__ == "__main__":

    filename = os.path.join(MODELDIR, "train_GAN.tfrecord")

    # # # create a GAN dataset
    # src_pth = "/mnt/lowe-sn00/Data/Alan/Anna_to_process/"
    # # src_pth = "/media/arl/DataII/Data/competition/RNN/"
    # # dirs = ['2017_02_28/pos0',
    # #         '2017_02_28/pos2',
    # #         '2017_02_28/pos4',
    # #         '2017_02_28/pos6',
    # #         '2017_02_28/pos8',
    # #         '2017_02_28/pos10',
    # #         '2017_02_28/pos12',
    # #         '2017_02_28/pos14',
    # #         '2017_03_31/pos7',
    # #         '2017_03_31/pos9',
    # #         '2017_03_31/pos11',
    # #         '2017_03_31/pos13',
    # #         '2017_03_31/pos15',
    # #         '2017_03_31/pos17']
    # dirs = ['2017_03_31/pos7',
    #         '2017_03_31/pos9',
    #         '2017_03_31/pos11',
    #         '2017_03_31/pos13',
    #         '2017_03_31/pos15',
    #         '2017_03_31/pos17']
    # channels = ["gfp", "rfp"]
    #
    # src = []
    # for d in dirs:
    #     channel_files = []
    #     for c in channels:
    #         channel = os.listdir(os.path.join(src_pth, d, c+"/"))
    #         channel = [cf for cf in channel if cf.startswith((c,c.upper()))]
    #         channel_files.append(os.path.join(src_pth, d, c, channel[0]))
    #     src.append([s for s in channel_files])
    #
    # print src
    #
    # create_GAN_tfrecord(src, filename, n_samples=256)

    # # get the configuration
    config = GAN2DConfiguration()
    config.batch_size = 32
    config.training_data = filename

    # set up the GAN
    mode = tf.estimator.ModeKeys.TRAIN
    gan = GenerativeAdverserialNetwork(config.to_params(), mode)

    gan.output_dir = "/mnt/lowe-sn00/JobServer/output/GAN/"

    gan.train()
