"""
Classes for parsing the structured files for the data

"""

import glob
import logging
import os


class VideoExtractorBase:
    """ The base class for the following video extractor classes """
    def __init__(self,
                 video_src_dir,
                 dir_level=2,
                 num_worker=8,
                 video_ext="mp4",
                 mixed_ext=False):

        self.video_src_dir = video_src_dir
        self.dir_level = dir_level
        self.num_worker = num_worker
        self.video_ext = video_ext  # support 'avi', 'mp4', 'webm'
        self.mixed_ext = mixed_ext

        #assert self.dir_level == 2  # we insist two-level data directory setting

        logging.info("Reading videos from folder: %s", self.video_src_dir)
        if self.mixed_ext:
            logging.info("Using the mixture extensions of videos")
            fullpath_list = glob.glob(self.video_src_dir +
                                      '/*' * self.dir_level)
        else:
            logging.info("Using the mixture extensions of videos: %s",
                         self.video_ext)
            fullpath_list = glob.glob(self.video_src_dir +
                                      '/*' * self.dir_level + '.' +
                                      self.video_ext)

        logging.info("Total number of videos found: %s", fullpath_list)

        # the full path list is the full path of the video,
        # for example: ./data/Kinetics/Kinetics700/train/video/clay_pottery_making/RE6YNPccYK4.mp4',
        self.fullpath_list = fullpath_list

        # Video item containing video full path,
        # for example: clay_pottery_making/RE6YNPccYK4.mp4
        self.videos_path_list = list(
            map(
                lambda p: os.path.join(os.path.basename(os.path.dirname(p)),
                                       os.path.basename(p)),
                self.fullpath_list))

    def organize_modality_dir(self, src_dir, to_dir):
        """ Organize the data dir of the modality into two level - calssname/data_id"""

        classes = os.listdir(src_dir)
        for classname in classes:
            new_dir = os.path.join(to_dir, classname)
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)
