"""
Tools for extracting information from the audio

"""

import glob
import os

from multiprocessing import Pool

from mmaction.tools.data import build_audio_features

from plato.datasources.datalib import modality_extraction_base


def obtain_audio_dest_dir(out_dir, audio_path):
    """ Get the destination path to save the audio """
    if '/' in audio_path:
        class_name = os.path.basename(os.path.dirname(audio_path))
        _, tail = os.path.split(audio_path)
        _ = tail.split(".")[0]  # should get the audio_name
        out_full_path = os.path.join(out_dir, class_name)
    else:  # the class name is not contained
        _ = audio_path.split(".")[0]  # audio_name
        out_full_path = out_dir

    return out_full_path


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
    except OSError:
        with open('extract_wav_err_file.txt', 'a+') as error_file:
            error_file.write(f'{line}\n')


class VideoAudioExtractor(modality_extraction_base.VideoExtractorBase):
    """ A class for extracting audio from the video """
    def __init__(self,
                 video_src_dir,
                 dir_level=2,
                 num_worker=2,
                 video_ext="mp4",
                 mixed_ext=False,
                 audio_ext="wav"):
        super().__init__(video_src_dir, dir_level, num_worker, video_ext,
                         mixed_ext)
        self.audio_ext = audio_ext

    def build_audios(
        self,
        to_dir,
    ):
        """ Extract audios in parallel """
        sourc_video_dir = self.video_src_dir
        if self.dir_level == 2:
            self.organize_modality_dir(src_dir=sourc_video_dir, to_dir=to_dir)
        _ = glob.glob(to_dir + '/*' * self.dir_level + '.wav')

        pool = Pool(self.num_worker)
        pool.map(
            extract_audio_wav,
            zip(self.fullpath_list,
                len(self.videos_path_list) * [sourc_video_dir],
                len(self.videos_path_list) * [to_dir]))

    def build_audios_features(
        self,
        audio_src_path,  # the dir that contains the src audio files
        to_dir,  # dir to save the extracted features
        frame_rate=30,  # The frame rate per second of the video.
        sample_rate=16000,  # The sample rate for audio sampling
        num_mels=80,  # Number of channels of the melspectrogram. Default
        fft_size=1280,  # fft_size / sample_rate is window size
        hop_size=320,  # hop_size / sample_rate is step size
        #spectrogram_type='lws',  # lws, 'librosa', recommand lws
        part="1/1"):
        """ Obtain the feature from the audio§§ """
        audio_tools = build_audio_features.AudioTools(frame_rate=frame_rate,
                                                      sample_rate=sample_rate,
                                                      num_mels=num_mels,
                                                      fft_size=fft_size,
                                                      hop_size=hop_size)

        audio_files = glob.glob(audio_src_path + '/*' * self.dir_level + '.' +
                                self.audio_ext)
        files = sorted(audio_files)

        if part is not None:
            [this_part, num_parts] = [int(i) for i in part.split('/')]
            part_len = len(files) // num_parts

        extractor_pool = Pool(self.num_worker)
        for file in files[part_len * (this_part - 1):(
                part_len *
                this_part) if this_part != num_parts else len(files)]:
            out_full_path = obtain_audio_dest_dir(out_dir=to_dir,
                                                  audio_path=file)

            extractor_pool.apply_async(
                build_audio_features.extract_audio_feature,
                args=(file, audio_tools, out_full_path))
        extractor_pool.close()
        extractor_pool.join()
