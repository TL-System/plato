"""
Tools for extracting and processing the frames
    The classes in this tool aim to extract different modalities,
    including rgb, optical flow, and audio
    from the raw video dataset.
"""

import glob
import os
from multiprocessing import Pool

from mmaction.tools.flow_extraction import extract_dense_flow

from plato.datasources.datalib import modality_extraction_base


def obtain_video_dest_dir(out_dir, video_path):
    """ Get the destination path for the video """
    if '/' in video_path:
        class_name = os.path.basename(os.path.dirname(video_path))
        head, tail = os.path.split(video_path)
        video_name = tail.split(".")[0]

        if class_name not in head:
            out_full_path = os.path.join(out_dir, class_name, video_name)
        else:
            out_full_path = os.path.join(out_dir, video_name)
    else:  # the class name is not contained
        video_name = video_path.split(".")[0]

        out_full_path = os.path.join(out_dir, video_name)

    return out_full_path


def extract_dense_flow_wrapper(items):
    """ This function can extract the frame based on the cpu hardware"""
    input_video_path, dest_dir, bound, save_rgb, start_idx, rgb_tmpl, flow_tmpl, method = items

    out_full_path = obtain_video_dest_dir(dest_dir, input_video_path)

    extract_dense_flow(input_video_path, out_full_path, bound, save_rgb,
                       start_idx, rgb_tmpl, flow_tmpl, method)


def extract_rgb_frame(videos_extraction_items):
    """Generate optical flow using dense flow.

    Args:
        videos_items (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, _, out_dir, new_width, new_height, new_short = videos_extraction_items
    out_full_path = obtain_video_dest_dir(out_dir=out_dir, video_path=vid_path)

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
    """ Extract optical flow from the video """
    full_path, vid_path, _, method, out_dir, new_short, new_width, new_height = videos_items
    out_full_path = obtain_video_dest_dir(out_dir=out_dir, video_path=vid_path)

    if new_short == 0:
        cmd = os.path.join(
            f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
            f' -nw={new_width} --nh={new_height} -v')
    else:
        cmd = os.path.join(
            f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
            f' -ns={new_short} -v')

    os.system(cmd)


class VideoFramesExtractor(modality_extraction_base.VideoExtractorBase):
    """ The class for extracting the frame the video """
    def __init__(self,
                 video_src_dir,
                 dir_level=2,
                 num_worker=8,
                 video_ext="mp4",
                 mixed_ext=False):
        super().__init__(video_src_dir, dir_level, num_worker, video_ext,
                         mixed_ext)

    def build_rgb_frames(self, to_dir, new_short=0, new_width=0, new_height=0):
        """ Obtain the RGB frame """
        sourc_video_dir = self.video_src_dir
        if self.dir_level == 2:
            self.organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        _ = glob.glob(to_dir + '/*' * self.dir_level)

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
            flow_type=None,  # None, 'tvl1', 'warp_tvl1', 'farn', 'brox',
            new_short=0,  # resize image short side length keeping ratio
            new_width=0,
            new_height=0):
        """ Get the optical flow frame based on the CPU """
        sourc_video_dir = self.video_src_dir
        if self.dir_level == 2:
            self.organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        _ = glob.glob(to_dir + '/*' * self.dir_level)

        pool = Pool(self.num_worker)
        pool.map(
            extract_optical_flow,
            zip(self.fullpath_list, self.videos_path_list,
                range(len(self.videos_path_list)),
                len(self.videos_path_list) * [flow_type],
                len(self.videos_path_list) * [to_dir],
                len(self.videos_path_list) * [new_short],
                len(self.videos_path_list) * [new_width],
                len(self.videos_path_list) * [new_height]))

    def build_frames_gpu(self,
                         rgb_out_dir_path,
                         flow_our_dir_path,
                         new_short=1,
                         new_width=0,
                         new_height=0):
        """ Get the optical flow frame based on the GPU  """
        self.build_rgb_frames(rgb_out_dir_path,
                              new_short=new_short,
                              new_width=new_width,
                              new_height=new_height)
        self.build_optical_flow_frames(flow_our_dir_path,
                                       new_short=new_short,
                                       new_width=new_width,
                                       new_height=new_height)

    def build_full_frames_gpu(self,
                              to_dir_path,
                              new_short=1,
                              new_width=0,
                              new_height=0):
        """ The interface from extracting all frames based on the GPU  """
        self.build_frames_gpu(rgb_out_dir_path=to_dir_path,
                              flow_our_dir_path=to_dir_path,
                              new_short=new_short,
                              new_width=new_width,
                              new_height=new_height)

    def build_frames_cpu(
        self,
        to_dir,
        bound=20,  # maximum of optical flow
        save_rgb=True,  # also save rgb frames
        start_idx=1,  # index of extracted frames
        rgb_tmpl="img_{:05d}.jpg",  # template filename of rgb frames
        flow_tmpl="{}_{:05d}.jpg",  # template filename of flow frames
        method="tvl1"):  # use which method to generate the flow
        """ Get the full frames, including RGB and optical flow based on the GPU """
        sourc_video_dir = self.video_src_dir
        if self.dir_level == 2:
            self.organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        _ = glob.glob(to_dir + '/*' * self.dir_level)

        pool = Pool(self.num_worker)
        pool.map(
            extract_dense_flow_wrapper,
            zip(self.fullpath_list,
                len(self.videos_path_list) * [to_dir],
                len(self.videos_path_list) * [bound],
                len(self.videos_path_list) * [save_rgb],
                len(self.videos_path_list) * [start_idx],
                len(self.videos_path_list) * [rgb_tmpl],
                len(self.videos_path_list) * [flow_tmpl],
                len(self.videos_path_list) * [method]))
