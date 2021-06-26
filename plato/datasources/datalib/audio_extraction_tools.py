#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import sys
import warnings
from multiprocessing import Pool

from mmaction.tools.data import build_audio_features
import numpy as np

from plato.datasources.datalib import modality_extraction_base


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


class VideoAudioExtractor(modality_extraction_base.VideoExtractorBase):
    def __init__(self,
                 video_src_dir,
                 dir_level=2,
                 num_worker=2,
                 video_ext="mp4",
                 mixed_ext=False):
        super().__init__(video_src_dir, dir_level, num_worker, video_ext,
                         mixed_ext)

    def build_audios(
        self,
        to_dir,
    ):
        sourc_video_dir = self.video_src_dir
        self.organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        done_fullpath_list = glob.glob(to_dir + '/*' * self.dir_level + '.wav')

        pool = Pool(self.num_worker)
        pool.map(
            extract_audio_wav,
            zip(self.fullpath_list,
                len(self.videos_path_list) * [sourc_video_dir],
                len(self.videos_path_list) * [to_dir]))