### Training data

Training data should be organised as follows:
```
set1/
  phase/
    0001_phase.tif
    0002_phase.tif
    ...
  gfp/
  rfp/
  ...
  label/
    0001_mask.tif
    0002_mask.tif
    ...
  weights/
    0001_weights.tif
    0002_weights.tif
    ...
set2/
```

Use the following scripts to generate weigthmaps and TFRecord files...

```bash
$ python weightmap.py
$ python generate_records.py
```


### Training weights

Per-pixel weighting can be applied during training to emphasise certain regions of images important for accurate segmentation. Below is an example of higher weighting in regions with
very few pixels separating objects:

[![weights](http://lowe.cs.ucl.ac.uk/images/labels_and_weights2.png)]()
*Object labels and pre-calculated weight maps*


### Typical job files

Train a classifier
```python
module = classifier
func = SERVER_train_classifier
device = GPU
params = {'path':'/media/arl/DataII/Data/competition/training/CNN_Training_Anna/png',
 'training_data': 'cell_cycle_CNN.tfrecord',
 'name':'CNN_competition',
 'num_inputs':2,
 'num_epochs':1000,
 'batch_size':32}
 ```

Run a classifier prediction:
```python
module = classifier
func = SERVER_classify_from_HDF
device = GPU
params = {'path': '/media/arl/DataII/Data/competition/colony',
 'image_dict': {'brightfield': 'brightfield/BF_pos11.tif',
                'gfp':'fluorescence/GFP_pos12.tif',
                'rfp':'fluorescence/RFP_pos11.tif'},
 'name':'CNN_competition'}
 ```
Train a UNet

```python
module = unet2d
func = SERVER_train_unet2d
device = GPU
params = {'path': '/media/arl/DataII/Data/AnaLisicia/training',
 'training_data': 'train_competition_UNet_w0-30.00_sigma-3.00.tfrecord',
 'name':'UNet2D_test',
 'shape':(1200,1600),
 'num_inputs':1,
 'num_outputs':2,
 'num_epochs':1000,
 'batch_size':1,
 'warm_start':False,
 'bridge':'concat'}
 ```

Run a UNet prediction:
```python
module = unet2d
func = SERVER_predict_unet2d
device = GPU
params = {'path': '/media/arl/DataII/Data/competition/colony',
 'image_dict': {'gfp':'fluorescence/GFP_pos12.tif',
                'rfp':'fluorescence/RFP_pos11.tif'},
 'name':'UNet2D_competition',
 'shape':(1200,1600),
 'num_inputs':2,
 'num_outputs':3}
 ```

Track cells
