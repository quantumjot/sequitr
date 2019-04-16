# Sequitr
**-- the conclusion of an inference**

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


---

### Documentation

Complete instructions for installation, training and running a server are found in the [wiki](https://github.com/quantumjot/sequitr/wiki).


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
