import os
import wget
import zipfile
import numpy as np

import park
from park.utils.misc import create_folder_if_not_exists


def get_chunk_time(trace, t_idx):
    if t_idx == len(trace[0]) - 1:
        return 1  # bandwidth last for 1 second
    else:
        return trace[0][t_idx + 1] - trace[0][t_idx]


def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    print(park.__path__[0])
    video_folder = park.__path__[0] + '/envs/abr_sim/videos/'
    create_folder_if_not_exists(video_folder)
    if not os.path.exists(video_folder + 'video_sizes.npy'):
        wget.download(
            'https://www.dropbox.com/s/hg8k8qq366y3u0d/video_sizes.npy?dl=1',
            out=video_folder + 'video_sizes.npy')

    chunk_sizes = np.load(video_folder + 'video_sizes.npy')

    return chunk_sizes


def load_traces():
    # download video size folder if not existed
    trace_folder = park.__path__[0] + '/envs/abr_sim/traces/'

    if not os.path.exists(trace_folder):
        wget.download(
            'https://www.dropbox.com/s/xdlvykz9puhg5xd/cellular_traces.zip?dl=1',
            out=park.__path__[0] + '/envs/abr_sim/')
        with zipfile.ZipFile(
             park.__path__[0] + '/envs/abr_sim/cellular_traces.zip', 'r') as zip_f:
            zip_f.extractall(park.__path__[0] + '/envs/abr_sim/')

    all_traces = []

    for trace in sorted(os.listdir(trace_folder)):
        
        all_t = []
        all_bandwidth = []

        with open(trace_folder + trace, 'r') as f:

            for line in f:
                parse = line.split()
                all_t.append(float(parse[0]))
                all_bandwidth.append(float(parse[1]))

        all_traces.append((all_t, all_bandwidth))

    return all_traces


def sample_trace(all_traces, np_random, trace_idx, test):
    # weighted random sample based on trace length
    all_p = [len(trace[1]) for trace in all_traces]
    sum_p = float(sum(all_p))
    all_p = [p / sum_p for p in all_p]
    if trace_idx is None:
        # sample a trace
        trace_idx = np_random.choice(len(all_traces), p=all_p)
        # sample a starting point	
        #init_t_idx = np_random.choice(len(all_traces[trace_idx][0]))	
        init_t_idx = 0	
    else:	
        all_p = [len(trace[1]) for trace in all_traces[:trace_idx+1]]	
        sum_p = float(sum(all_p))	
        all_p = [p / sum_p for p in all_p]	
        	
        if not test:	
            trace_idx = np_random.choice(len(all_traces[:trace_idx+1]), p=all_p)	
        	
        init_t_idx = 0
    # return a trace and the starting t
    return all_traces[trace_idx], init_t_idx
