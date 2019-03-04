# Sequitr
** -- the conclusion of an inference**

**WORK IN PROGRESS** (Last update: 04/03/2019)  
*Please note, this is not the full repository (yet)*


*Sequitr* is a small, lightweight Python library for common image processing tasks in optical microscopy, in particular, single-molecule imaging, super-resolution or
time-lapse imaging of cells.

[![conv-net-output](http://lowe.cs.ucl.ac.uk/images/segmentation.png)]()  
*Example of segmenting and localizing cells in low contrast microscopy images*

*Sequitr* works in conjunction with BayesianTracker (btrack,
https://github.com/quantumjot/BayesianTracker) for microscopy data analysis. For
 more information see: http://lowe.cs.ucl.ac.uk/

---

### Dependencies

*Sequitr* has been tested with Python 2.7+ on OS X and Linux, and requires
the following additional packages:

+ TensorFlow
+ Numpy
+ Scipy
+ h5py
+ Scikit-Image
+ Christoph Gohlke's tifffile.py (https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html, needs to be added to /dataio folder)

*Sequitr* is written in Python. For best performance we recommend using a GPU with at least 8Gb RAM.



### Using the Generative Adversarial Network

First generate the TFRecord files of the images to sample from. The n_samples parameter choses n_samples frames from the original dataset:
```python
import gan

src = [["<some_path_to_your_data>/gfp.tif", "<some_path_to_your_data>/rfp.tif"]]
filename = "train_GAN.tfrecord"
gan.create_GAN_tfrecord(src, filename, n_samples=256)
```

Then, to train the network:
```python
# set up a GAN configuration
config = gan.GAN2DConfiguration()
config.batch_size = 32
config.training_data = "train_GAN.tfrecord"

# set up the GAN
mode = tf.estimator.ModeKeys.TRAIN
GAN = gan.GenerativeAdverserialNetwork(config.to_params(), mode)
GAN.output_dir = "<some_path_to_your_output>/GAN/"

# train it
GAN.train()
```

There are lots of other parameters to play with. By default this creates the discriminator and generator networks on different GPUs. You may need to change that. Once training, the stdout looks like this:
```
Building new graph...
Building <function generator at 0x7fa6bd55cc08> on dev:/gpu:1
Building <function discriminator at 0x7fa6bd55ccf8> on dev:/gpu:0
Building <function discriminator at 0x7fa6bd55ccf8> on dev:/gpu:0
tensorflow/core/kernels/data/shuffle_dataset_op.cc:98] Filling up shuffle buffer (this may take a while): 339 of 1536
tensorflow/core/kernels/data/shuffle_dataset_op.cc:98] Filling up shuffle buffer (this may take a while): 728 of 1536
tensorflow/core/kernels/data/shuffle_dataset_op.cc:98] Filling up shuffle buffer (this may take a while): 1112 of 1536
tensorflow/core/kernels/data/shuffle_dataset_op.cc:98] Filling up shuffle buffer (this may take a while): 1505 of 1536
tensorflow/core/kernels/data/shuffle_dataset_op.cc:136] Shuffle buffer filled.
Alpha: 0.00, Exp: 0, Sz: (4, 4), Step: 0/ 6720, D_loss: 0.69923, G_loss: 0.91889, Model size: 14
Alpha: 0.02, Exp: 0, Sz: (4, 4), Step: 10/ 6720, D_loss: 0.50238, G_loss: 1.52228, Model size: 14
Alpha: 0.04, Exp: 0, Sz: (4, 4), Step: 20/ 6720, D_loss: 0.51882, G_loss: 0.94966, Model size: 14
Alpha: 0.06, Exp: 0, Sz: (4, 4), Step: 30/ 6720, D_loss: 0.38884, G_loss: 2.19388, Model size: 14
Alpha: 0.08, Exp: 0, Sz: (4, 4), Step: 40/ 6720, D_loss: 0.47636, G_loss: 2.56428, Model size: 14
Alpha: 0.10, Exp: 0, Sz: (4, 4), Step: 50/ 6720, D_loss: 0.47226, G_loss: 2.78979, Model size: 14
Alpha: 0.13, Exp: 0, Sz: (4, 4), Step: 60/ 6720, D_loss: 0.24605, G_loss: 2.47885, Model size: 14
...
```

### Launching the server

On the first run, you can either manually create a 'server.config' file or run:

```bash
$ python server.py --setup
```

This will execute the auto configuration of the server, creating a list of TF compatible GPU and CPU compute devices.  The server can then be started using the following command:

```bash
$ python server.py
```

The following (optional) flags can be used to specify how the server instance is configured.

```
usage: server.py [-h] [--jobdir JOBDIR] [--logdir LOGDIR]
                 [--gpus [{0,1,2,3,4,5,6,7} [{0,1,2,3,4,5,6,7} ...]]]
                 [--local] [--setup] [--use_config USE_CONFIG]

Sequitr server process

optional arguments:
  -h, --help            show this help message and exit
  --jobdir JOBDIR       Path to job directory
  --logdir LOGDIR       Path to log directory
  --gpus [{0,1,2,3,4,5,6,7} [{0,1,2,3,4,5,6,7} ...]]
                        Specify the gpus which can be used for processing
  --local               Running a local server only. Prevents pinging.
  --setup               Runs the setup configuration to determine hardware
                        specs, and generate the config file. On server
                        restart, the config will persist.
  --use_config USE_CONFIG
                        Use a specific (non-default) server config file. The
                        default option, without specifying the --config flag
                        is server.config. This is the file written out when
                        the --setup flag has previously been used to create a
                        server configuation.
```
