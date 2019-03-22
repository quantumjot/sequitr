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

# import impy
import numpy as np
import os
import re
import time
import matplotlib.pyplot as plt




OCTOPUS_FILE_TYPES = ["uint8","uint16"]


class OctopusData(object):
    """
    OctopusData

    An object for reading large contiguous chunks of octopus data.  Takes in the
    file name and returns slices of a numpy array containing the image data.
    Header data can also be returned for access to nanopositioner or other
    device data.

    Has an option to use contiguous blocks of data or not.  Can also use
    OctopusMerge function to generate new contiguous files.


    Args:
        filename: path and stem name of the octopus stream
        verbose: (bool) display extra information (default: True)
        contiguous: (bool) treat non-contiguos blocks of data (e.g. M_1, M_5)..
                     with the same stem (M_) as one file (default: True)
        header: (bool) only load the headers

    Properties:
        bit_depth: bit depth of the images (typically uint8 or uint16)
        header_keys: names of the headers

    Member functions:
        refresh: refreshes the files belonging to the stream. Useful for server
            processing of data coming from instruments


    Notes:
        OctopusData[n] Return impyImage of frame n from the stack

    """

    def __init__(self,
                 filename,
                 contiguous=True,
                 header=False,
                 verbose=False):
        """ Initialise the object, input is a filename and first and last files
        of the octopus range.  No assumptions are made about the number of
        frames per file.  We will also gather all header information together
        for use too.
        """

        self.data = None
        self.fileopen = -1
        self.framesize = -1
        self.filenum_to_framerange = {}
        self.currentfile = -1
        self.use_contig = contiguous
        self.__header_only = header
        self.__verbose = verbose

        # set up some params
        self.filename = filename
        self.filelist = []
        self.num_frames = 0
        self.__header_keys = {}
        self.timeout = 60

        # refresh the file list
        self.refresh()

        # now load the first file, get some parameters and set them
        self.__open_header(self.filename + str(self.filelist[0]))
        self.framesize = (int(self.header(0)['H']),int(self.header(0)['W']))

        if 'Bit_Depth' in self.header(0):
            self.__bit_depth = int(self.header(0)['Bit_Depth'])
        else:
            self.__bit_depth = 16


        if self.__verbose:
            print 'Opened Octopus data file, size {0:d}x{1:d}, {2} frames \
                ({3:d}-bit)...'.format(self.framesize[1],self.framesize[0], \
                self.num_frames, self.bit_depth)

    @property
    def bit_depth(self): return self.__bit_depth

    @property
    def header_keys(self):
        return self.__header_keys

    def header(self, frame_num):
        return self.__return_header(frame_num)

    def __find_file_range(self):
        """ With a given file name, parse out the useful info and return a list
        of file numbers for the given stem. When used in contiguous mode, this
        searches for consecutive files and only returns the first set of this
        list.
        """
        datadir, stem = os.path.split(self.filename)
        self.filestem = stem

        # check that we have a valid data directory
        try:
            files = os.listdir(datadir)
        except IOError:
            raise IOError('No files exist in directory: {0:s}'.format(datadir))

        filenums = []
        for file in files:
            # strip off the stem, and get the number
            targetfile = re.match(stem+'([0-9]*)\.dth', file)
            # only take thosefiles which match the formatting requirements
            if targetfile: filenums.append( int(targetfile.group(1)) )


        if not filenums:
            raise IOError('No Octopus stream with pattern {0:s} found.'.format(stem))

        # sort the file numbers
        sfilenums = sorted(filenums)

        # if we are in contiguous mode, get the first contiguous block,
        # else use the whole list
        if self.use_contig:
            filelist = []
            filelist.append(sfilenums[0])
            for i in xrange(1,len(sfilenums)):
                if (sfilenums[i] == sfilenums[i-1]+1):
                    filelist.append(sfilenums[i])
                else:
                    break
        elif not self.use_contig:
            filelist = sfilenums

        # print out some info here
        if self.__verbose:
            print 'Found files {0:s}{1:d} to {2:d}...'.format(stem, np.min(filelist), np.max(filelist))

        return filelist

    def __getitem__(self, abs_frame_num):
        """ Main function, takes an absoulte frame number and returns the image
        data from the octopus files
        """
        frame, info = self.__absolute_frame(abs_frame_num)
        return frame

    def __absolute_frame(self, abs_frame_num):
        """ Major function, this returns the frame data as a numpy array"""

        frame = np.array(())
        # first check to see whether we have a current file
        if (self.currentfile != -1):
            # now is the current file in a useful range
            j = self.currentfile
            if (abs_frame_num >= self.filenum_to_framerange[j][0] and abs_frame_num <= self.filenum_to_framerange[j][1]):
                if not self.__header_only:
                    frame = self.__return_frame( abs_frame_num - self.filenum_to_framerange[j][0] )
                info = self.__return_header( abs_frame_num - self.filenum_to_framerange[j][0] )
                info['N'] = abs_frame_num
                return frame, info

        # last ditch effort, file is not open, and we need to find it:
        for i in self.filelist:
            # find the appropriate file number from the dictionary
            if (abs_frame_num >= self.filenum_to_framerange[i][0] and abs_frame_num <= self.filenum_to_framerange[i][1]):
                # now go ahead and open the appropriate file
                self.currentfile = i
                if not self.__header_only:
                    self.__open_file_and_header(self.filename + str(i))
                    frame = self.__return_frame( abs_frame_num - self.filenum_to_framerange[i][0] )
                else:
                    self.__open_header(self.filename + str(i))

                info = self.__return_header( abs_frame_num - self.filenum_to_framerange[i][0] )
                info['N'] = abs_frame_num
                return frame, info

    def __open_file_and_header(self, filename):
        """ Function to open both file and header. """
        self.__open_header(filename)
        self.__open_file(filename, len(self.__header))

    def __open_header(self, filename):
        """ Function to open only the octopus header file. """
        try:
            fh = open(filename+'.dth')
            fp_header = fh.readlines()
        except IOError:
            fp_header = []
            raise IOError( filename + " is not a valid file" )
        # parse the header
        self.__header = [re.findall('\S+:\s*(\S+)',line) for line in fp_header]

        # grab a list of header parameters that are saved, we can use these as
        # keys for the 'info' dictionary that is appended to any image
        self.__header_keys = re.findall('(\w*)\s*:\s*',fp_header[0])
        fh.close()

    def __open_file(self, filename, num_frames):
        """ Function to open only the octopus data file as a memmap object """
        try:
            self.data = np.memmap(filename+'.dat', dtype='uint'+str(self.bit_depth), mode='r', shape=(num_frames,self.framesize[0],self.framesize[1]))
        except IOError:
            self.data = []
            raise IOError( filename + " is not a valid file. Make sure the path to the images still exists!" )
        self.fileopen = True

    def __return_frame(self, frame_num):
        """ Returns a single numpy array of floats, with shape self.size
        corresponding to frame_num in the current open file. This does not use
        the magical single frame index! Use __getitem__ for that!
        """
        return np.array(self.data[frame_num,:,:], dtype='float')

    def __return_header(self, frame_num):
        """  Returns header info for the frame.

        Old style:
        return np.array(self.__header[frame_num][0:7], dtype='float')

        New style:
        return dictionary with header keys {'N':0, 'X':512,...}
        """
        return dict((self.__header_keys[i], self.__header[frame_num][i]) for i in xrange(len(self.__header_keys)))

    def __len__(self):
        """ Return the number of frames of the movie. """
        return self.num_frames

    def save_as_PNG(self, output_dir, start_frame=0):
        raise DeprecationWarning('Warning. Save_as_PNG has been deprecated')

    def refresh(self):
        """ Refresh the file list and update the frame mapping.

        This function is useful for dealing with data coming directly from
        instruments. It will update the Octopus stream as new files become
        available.

        Returns:
            True or False based on whether new files were found...

        Notes:
            ARL 2015/08/27 - Modified this to check when the last modification
            of the file occurred. If it was too recent, ignore from the current
            update.
        """

        to_update = []

        new_filelist = self.__find_file_range()
        for new_file in new_filelist:
            last_modified = os.stat(self.filename + str(new_file)+ ".dth").st_mtime
            if new_file not in self.filelist and (time.time()-last_modified)>self.timeout:
                to_update.append(new_file)


        if not to_update: return False

        # now loop through all of the headers and append to working filelist
        for new_file in to_update:
            self.filelist.append(new_file)
            self.__open_header(self.filename + str(new_file))
            num_frames_in_file = len(self.__header)
            self.filenum_to_framerange[new_file] = (self.num_frames, self.num_frames+num_frames_in_file-1)
            self.num_frames += num_frames_in_file

        if self.__verbose:
            print "Updated Octopus stream with {0:d} new files...".format(len(to_update))

        return True


    def to_array(self):
        """ Return the entire stack as an array """
        image_data = np.zeros((len(self),self.framesize[0], self.framesize[1]), dtype='uint8')
        for i in xrange(len(self)):
            image_data[i,...] = self[i]
        return image_data





if __name__ == '__main__':

    root_pth = '/media/lowe-sn00/Data/Anna/back-up/Competition in dishes_new lines_2016/17_02_28'
    folders = ['re-start','New folder','restart2','restart3']
    images = ['BF_pos0_','RFP_pos0_','GFP_pos1_',
              'BF_pos2_','RFP_pos2_','GFP_pos3_',
              'BF_pos4_','RFP_pos4_','GFP_pos5_',
              'BF_pos6_','RFP_pos6_','GFP_pos7_',
              'BF_pos8_','RFP_pos8_','GFP_pos9_',
              'BF_pos10_','RFP_pos10_','GFP_pos11_',
              'BF_pos12_','RFP_pos12_','GFP_pos13_',
              'BF_pos14_','RFP_pos14_','GFP_pos15_']

    out_dir = '/home/arl/Documents/Data/Anna/17_02_28'

    import tifffile as t

    for image in images:
        for file_part, folder in enumerate(folders):

            pth = os.path.join(root_pth, *folders[:file_part+1])
            print pth
            fn = os.path.join(pth, image)
            o = OctopusData(fn, contiguous=False)
            print o.framesize
            odata = o.to_array()
            out_fn = os.path.join(out_dir, image+'part'+str(file_part)+'.tif')

            print image, folder, out_fn
            t.imsave(out_fn, odata)
