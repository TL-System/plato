#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import sys
import warnings
from multiprocessing import Pool

from mmaction.tools.data import build_rawframes, build_audio_features
import numpy as np
"""
    The classes in this tool aim to extract different modalities, including rgb, optical flow, and audio 
from the raw video dataset. 
"""


def extract_rgb_frame(videos_extraction_items):
    """Generate optical flow using dense flow.

    Args:
        videos_items (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id, out_dir, new_width, new_height, new_short = videos_extraction_items
    if '/' in vid_path:
        act_name = os.path.basename(os.path.dirname(vid_path))
        out_full_path = os.path.join(out_dir, act_name)
    else:
        out_full_path = out_dir

    if new_short == 0:
        cmd = os.path.join(
            f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
            f' -nw={new_width} -nh={new_height} -v')
    else:
        cmd = os.path.join(
            f"denseflow '{full_path}' -b=20 -s=0 -o='{out_full_path}'"
            f' -ns={new_short} -v')
    os.system(cmd)


def extract_optical_flow(videos_items):

    full_path, vid_path, vid_id, method, task, out_dir, new_width, new_height, new_short = videos_items

    if '/' in vid_path:
        act_name = os.path.basename(os.path.dirname(vid_path))
        out_full_path = os.path.join(out_dir, act_name)
    else:
        out_full_path = out_dir

    if new_short == 0:
        cmd = os.path.join(
            f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
            f' -nw={new_width} --nh={new_height} -v')
    else:
        cmd = os.path.join(
            f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
            f' -ns={new_short} -v')

    os.system(cmd)


def extract_audio_wav(line_times):
    """Extract the audio wave from video streams using FFMPEG."""
    line, root, out_dir = line_times
    video_id, _ = os.path.splitext(os.path.basename(line))
    video_dir = os.path.dirname(line)
    video_rel_dir = os.path.relpath(video_dir, root)
    dst_dir = os.path.join(out_dir, video_rel_dir)
    os.popen(f'mkdir -p {dst_dir}')
    try:
        if os.path.exists(f'{dst_dir}/{video_id}.wav'):
            return
        cmd = f'ffmpeg -i ./{line}  -map 0:a  -y {dst_dir}/{video_id}.wav'
        os.popen(cmd)
    except BaseException:
        with open('extract_wav_err_file.txt', 'a+') as f:
            f.write(f'{line}\n')


class VideoModalityExtractor(object):
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

        assert self.dir_level == 2  # we insist two-level data directory setting

        logging.info(
            ("Reading videos from folder: {}").format(self.video_src_dir))
        if self.mixed_ext:
            logging.info("Using the mixture extensions of videos")
            fullpath_list = glob.glob(self.video_src_dir +
                                      '/*' * self.dir_level)
        else:
            logging.info(("Using the mixture extensions of videos: {}").format(
                self.video_ext))
            fullpath_list = glob.glob(self.video_src_dir +
                                      '/*' * self.dir_level + '.' +
                                      self.video_ext)

        logging.info(
            ("Total number of videos found: {}").format(len(fullpath_list)))

        self.fullpath_list = fullpath_list
        # Video item containing video full path,
        self.videos_path_list = list(
            map(
                lambda p: os.path.join(os.path.basename(os.path.dirname(p)),
                                       os.path.basename(p)),
                self.fullpath_list))

    def _organize_modality_dir(self, src_dir, to_dir):
        """ Organize the data dir of the modality into two level - calssname/data_id"""
        classes = os.listdir(src_dir)
        for classname in classes:
            new_dir = os.path.join(to_dir, classname)
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)

    def build_rgb_frames(self, to_dir, new_short=0, new_width=0, new_height=0):
        sourc_video_dir = self.video_src_dir
        self._organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        done_fullpath_list = glob.glob(to_dir + '/*' * self.dir_level)

        pool = Pool(self.num_worker)
        pool.map(
            extract_rgb_frame,
            zip(self.fullpath_list, self.videos_path_list,
                range(len(self.videos_path_list)),
                len(self.videos_path_list) * [to_dir],
                len(self.videos_path_list) * [new_short],
                len(self.videos_path_list) * [new_width],
                len(self.videos_path_list) * [new_height]))

    def build_optical_flow_frames(
            self,
            to_dir,
            new_short=0,  # resize image short side length keeping ratio
            new_width=0,
            new_height=0):
        sourc_video_dir = self.video_src_dir
        self._organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        done_fullpath_list = glob.glob(to_dir + '/*' * self.dir_level)

        pool = Pool(self.num_worker)
        pool.map(
            extract_optical_flow,
            zip(self.fullpath_list, self.videos_path_list,
                range(len(self.videos_path_list)),
                len(self.videos_path_list) * [flow_type],
                len(self.videos_path_list) * [task],
                len(self.videos_path_list) * [new_short],
                len(self.videos_path_list) * [new_width],
                len(self.videos_path_list) * [new_height]))

    def build_audios(
        self,
        to_dir,
    ):
        sourc_video_dir = self.video_src_dir
        self._organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        done_fullpath_list = glob.glob(to_dir + '/*' * self.dir_level + '.wav')

        pool = Pool(self.num_worker)
        pool.map(
            extract_audio_wav,
            zip(self.fullpath_list,
                len(self.videos_path_list) * [sourc_video_dir],
                len(self.videos_path_list) * [to_dir]))