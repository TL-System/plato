"""

Cut the whole video based on the requirements

data_root = '../../../data/gym'
video_root = f'{data_root}/videos'
anno_root = f'{data_root}/annotations'
anno_file = f'{anno_root}/annotation.json'

event_anno_file = f'{anno_root}/event_annotation.json'
event_root = f'{data_root}/events'

"""

import os
import os.path as osp
import subprocess

import mmcv


def trim_event(video_root, anno_file, event_anno_file, event_root):
    """ Trim the videos into many events """
    videos = os.listdir(video_root)
    videos = set(videos)
    annotation = mmcv.load(anno_file)
    event_annotation = {}

    mmcv.mkdir_or_exist(event_root)

    for anno_key, anno_value in annotation.items():
        if anno_key + '.mp4' not in videos:
            print(f'video {anno_key} has not been downloaded')
            continue

        video_path = osp.join(video_root, anno_key + '.mp4')

        for event_id, event_anno in anno_value.items():
            timestamps = event_anno['timestamps'][0]
            start_time, end_time = timestamps
            event_name = anno_key + '_' + event_id

            output_filename = event_name + '.mp4'

            command = [
                'ffmpeg', '-i',
                '"%s"' % video_path, '-ss',
                str(start_time), '-t',
                str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy',
                '-threads', '8', '-loglevel', 'panic',
                '"%s"' % osp.join(event_root, output_filename)
            ]
            command = ' '.join(command)
            try:
                subprocess.check_output(command,
                                        shell=True,
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(
                    f'Trimming of the Event {event_name} of Video {anno_key} Failed',
                    flush=True)

            segments = event_anno['segments']
            if segments is not None:
                event_annotation[event_name] = segments

    mmcv.dump(event_annotation, event_anno_file)


# data_root = '../../../data/gym'
# anno_root = f'{data_root}/annotations'

# event_anno_file = f'{anno_root}/event_annotation.json'
# event_root = f'{data_root}/events'
# subaction_root = f'{data_root}/subactions'


def trim_subsection(event_anno_file, event_root, subaction_root):
    """ Further trim the event into several subsections """
    events = os.listdir(event_root)
    events = set(events)
    annotation = mmcv.load(event_anno_file)

    mmcv.mkdir_or_exist(subaction_root)

    for anno_key, anno_value in annotation.items():
        if anno_key + '.mp4' not in events:
            print(f'video {anno_key[:11]} has not been downloaded '
                  f'or the event clip {anno_key} not generated')
            continue

        video_path = osp.join(event_root, anno_key + '.mp4')

        for subaction_id, subaction_anno in anno_value.items():
            timestamps = subaction_anno['timestamps']
            start_time, end_time = timestamps[0][0], timestamps[-1][1]
            subaction_name = anno_key + '_' + subaction_id

            output_filename = subaction_name + '.mp4'

            command = [
                'ffmpeg', '-i',
                '"%s"' % video_path, '-ss',
                str(start_time), '-t',
                str(end_time - start_time), '-c:v', 'libx264', '-c:a', 'copy',
                '-threads', '8', '-loglevel', 'panic',
                '"%s"' % osp.join(subaction_root, output_filename)
            ]
            command = ' '.join(command)
            try:
                subprocess.check_output(command,
                                        shell=True,
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                print(
                    f'Trimming of the Subaction {subaction_name} of Event '
                    f'{anno_key} Failed',
                    flush=True)


def generate_splits_list(data_root, annotation_root, frame_data_root):
    """ Generate the split information based on the predefined files """
    videos = os.listdir(data_root)
    videos = set(videos)
    train_file_org = osp.join(annotation_root, 'gym99_train_org.txt')
    val_file_org = osp.join(annotation_root, 'gym99_val_org.txt')
    train_file = osp.join(annotation_root, 'gym99_train.txt')
    val_file = osp.join(annotation_root, 'gym99_val.txt')
    train_frame_file = osp.join(annotation_root, 'gym99_train_frame.txt')
    val_frame_file = osp.join(annotation_root, 'gym99_val_frame.txt')

    train_org = open(train_file_org).readlines()

    train_org = [x.strip().split() for x in train_org]

    train = [x for x in train_org if x[0] + '.mp4' in videos]

    if osp.exists(frame_data_root):
        train_frames = []
        for line in train:
            length = len(os.listdir(osp.join(frame_data_root, line[0])))
            train_frames.append([line[0], str(length // 3), line[1]])
        train_frames = [' '.join(x) for x in train_frames]
        with open(train_frame_file, 'w') as fout:
            fout.write('\n'.join(train_frames))

    train = [x[0] + '.mp4 ' + x[1] for x in train]

    with open(train_file, 'w') as fout:
        fout.write('\n'.join(train))

    val_org = open(val_file_org).readlines()
    val_org = [x.strip().split() for x in val_org]
    val = [x for x in val_org if x[0] + '.mp4' in videos]

    if osp.exists(frame_data_root):
        val_frames = []
        for line in val:
            if not os.path.exists(osp.join(frame_data_root, line[0])):
                continue
            length = len(os.listdir(osp.join(frame_data_root, line[0])))
            val_frames.append([line[0], str(length // 3), line[1]])
        val_frames = [' '.join(x) for x in val_frames]
        with open(val_frame_file, 'w') as fout:
            fout.write('\n'.join(val_frames))

    val = [x[0] + '.mp4 ' + x[1] for x in val]
    with open(val_file, 'w') as fout:
        fout.write('\n'.join(val))
