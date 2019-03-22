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
import re

import numpy as np
import tifffile as t
import json
from datetime import datetime


class MicromanagerMetadataParser(object):
    """ Parse the micromanager metadata for a particular position and channel.

        Image filenames: img_channel000_position001_time000000002_z000.tif

    Args:
        filepath:   the folder containing the image data and metadata
        channel:    give a particular channel to return the data

    Properties:
        x_pos
        y_pos
        z_pos
        timestamps
        image_filenames

    Notes:
        - This works for micro-manager 2.0.

    TODO:
        (arl) Need to make this work across multiple positions/channels

    """


    def __init__(self, filepath, channel=None):

        root, position = os.path.split(filepath)
        filename = os.path.join(filepath, 'metadata.txt')

        try:
            with open(filename, 'r') as metadata_file:
                self.raw = json.load(metadata_file)
        except IOError as io:
            raise io

        if not channel:
            self.is_channel = lambda x: True
        else:
            assert isinstance(channel, int)
            self.is_channel = lambda x: x['ChannelIndex'] == channel

        self.pos_str = position+'/'
        self.root_str = root

    @property
    def summary(self):
        return self.raw['Summary']

    @property
    def coords(self):
        r = [self.raw[m] for m in self.raw.keys() if m.startswith('Coords')]
        r = [m for m in r if self.is_channel(m)]
        return sorted(r, key=lambda k: k['Frame'])

    @property
    def metadata(self):
        r = [self.raw[m] for m in self.raw.keys() if m.startswith('Meta')]
        r = [m for m in r if self.is_channel(m)]
        return sorted(r, key=lambda k: k['Frame'])

    @property
    def x_pos(self):
        return [r['XPositionUm'] for r in self.metadata]

    @property
    def y_pos(self):
        return [r['YPositionUm'] for r in self.metadata]

    @property
    def z_pos(self):
        return [r['ZPositionUm'] for r in self.metadata]

    @property
    def image_filenames(self):
        return [r['FileName'].replace(self.pos_str, '') for r in self.metadata]

    @property
    def timestamps(self):
        t_str = [r['ReceivedTime'] for r in self.metadata]
        t_sec = [self.convert_time_to_epoch(s) for s in t_str]
        return t_sec

    @property
    def shape(self):
        """ Return the shape of the stack: (frames, slices, width, height) """
        frames = self.summary['Frames']
        # positions = self.summary['Positions']
        # channels = self.summary['Channels']
        slices = self.summary['Slices']
        width = self.metadata[0]['Width']
        height = self.metadata[0]['Height']

        return (frames, slices, width, height)

    @property
    def start_time(self):
        start_time_str = self.summary['StartTime']
        return self.convert_time_to_epoch(start_time_str)

    @staticmethod
    def convert_time_to_epoch(time_str):
        """ Micromanager time format: 2019-03-15 18:44:35.225 +0000 """
        # need to strip the utc offset for Python 2+
        utc_time = datetime.strptime(time_str[:23], "%Y-%m-%d %H:%M:%S.%f")
        return (utc_time - datetime(1970, 1, 1)).total_seconds()

    def get_metadata(self, idx):
        m = {'filename': self.image_filenames[idx],
             'x_position': self.x_pos[idx],
             'y_position': self.y_pos[idx],
             'z_position': self.z_pos[idx],
             'timestamp': self.timestamps[idx]}
        return m







class MicromanagerReader_LEGACY(object):
    """ Reads in micromanager stacks

    Filename format:
        img_000000000_Brightfield_000.tif


    TODO(arl):
        - Read metadata and provide interface to read
    """

    def __init__(self, filepath, channel=None):

        raise DeprecationWarning('This will be deprecated soon')

        # if we didn't specify a channel, raise an error
        if channel is None:
            raise ValueError('A channel must be specified')

        # set the filepath
        self._dir = filepath

        # list all of the files in a directory, trim to only those of the
        # correct channel
        _files = os.listdir(filepath)
        _tiffs = [f for f in _files if f.endswith('.tif')]

        if len(_tiffs) < 1:
            raise IOError('No images found')

        files = [f for f in _tiffs if self._parse(f)['channel']==channel]
        files = sorted(files, key=lambda f: self._parse(f)['frame_number'])

        # store the sorted list of filenames
        self._files = files

        # set the dtype
        first_image = self[0]
        self._dtype = first_image.dtype

        # set the channel, width and height
        self.channel = channel
        self._w, self._h = first_image.shape

    @property
    def width(self): return self._w

    @property
    def height(self): return self._h

    def _parse(self, filename):
        p = 'img_(?P<framenum>[0-9]+)_(?P<channel>[a-zA-z]+)_(?P<slice>[0-9]+)'
        file_details = re.search(p, filename)

        deets = {'frame_number': int(file_details.group('framenum')),
                'channel': file_details.group('channel'),
                'slice': int(file_details.group('slice'))}

        return deets

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        assert(idx>=0 and idx<=len(self))
        image_filename = os.path.join(self._dir, self._files[idx])
        image = t.TIFFfile(image_filename).asarray()
        return image

    @property
    def dtype(self):
        return self._dtype







class MicromanagerReader(object):
    """ Reads in micromanager stacks """

    def __init__(self, filepath, channel=None):
        self.metadata = MicromanagerMetadataParser(filepath, channel)
        self._dir = filepath
        self._files = self.metadata.image_filenames

        self._n, self._s, self._w, self._h = self.metadata.shape

    @property
    def width(self): return self._w

    @property
    def height(self): return self._h

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        assert(idx>=0 and idx<=len(self))
        image_filename = os.path.join(self._dir, self._files[idx])
        image = t.TIFFfile(image_filename).asarray()
        return image

    def get_metadata(self, idx):
        assert(idx>=0 and idx<=len(self))
        return self.metadata.get_metadata(idx)

    @property
    def dtype(self):
        return self._dtype





if __name__ == '__main__':

    fp = '/home/arl/Dropbox/Data/encoder_test_4/Pos0'
    data = MicromanagerReader(filepath=fp, channel=0)

    print data[0], data.get_metadata(0)
