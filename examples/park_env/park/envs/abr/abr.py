# /usr/bin/python
# python 2.7 only because of protobuf

import os
import sys
import time
import json
import wget
import base64
import urllib
import socket
import string
import zipfile
import subprocess
import threading
import numpy as np
from sys import platform
from urllib.request import urlopen
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import UnixStreamServer

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding


S_INFO = 6  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42

# video chunk sizes
size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]


def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 8 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0: size_video6[index]}
    return sizes[quality]


class UnixHTTPServer(HTTPServer):
    address_family = socket.AF_UNIX

    def __init__(self, *args, **kwargs):
        HTTPServer.__init__(self, *args, **kwargs)
        self.stop = False

    def server_bind(self):
        UnixStreamServer.server_bind(self)
        self.server_name = "unix_socket_server"
        self.server_port = 0

    def serve_forever(self):
        while not self.stop:
            self.handle_request()


def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.agent = input_dict['agent']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            print('post_data', post_data)

            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                print("Summary: ", post_data)
            else:
                # option 1. reward for just quality
                # reward = post_data['lastquality']
                # option 2. combine reward for quality and rebuffer time
                #           tune up the knob on rebuf to prevent it more
                # reward = post_data['lastquality'] - 0.1 * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf'])
                # option 3. give a fixed penalty if video is stalled
                #           this can reduce the variance in reward signal
                # reward = post_data['lastquality'] - 10 * ((post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) > 0)

                # option 4. use the metric in SIGCOMM MPC paper
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])

                # --linear reward--
                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                        - REBUF_PENALTY * rebuffer_time / M_IN_K \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K

                # --log reward--
                # log_bit_rate = np.log(VIDEO_BIT_RATE[post_data['lastquality']] / float(VIDEO_BIT_RATE[0]))
                # log_last_bit_rate = np.log(self.input_dict['last_bit_rate'] / float(VIDEO_BIT_RATE[0]))

                # reward = log_bit_rate \
                #          - 4.3 * rebuffer_time / M_IN_K \
                #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

                # --hd reward--
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]

                # compute bandwidth measurement
                video_chunk_fetch_time = max(post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime'], 1)
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_count']
                self.input_dict['video_chunk_count'] += 1

                next_video_chunk_sizes = []
                for i in range(A_DIM):
                    next_video_chunk_sizes.append(get_chunk_size(i, self.input_dict['video_chunk_count']))

                # construct the latest observation for the agent
                obs = np.array([
                    VIDEO_BIT_RATE[post_data['lastquality']],
                    post_data['buffer'],
                    float(video_chunk_size) / float(video_chunk_fetch_time) / M_IN_K,  # mega byte / sec
                    float(video_chunk_fetch_time) / M_IN_K,  # sec
                    np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP),
                    next_video_chunk_sizes[0],
                    next_video_chunk_sizes[1],
                    next_video_chunk_sizes[2],
                    next_video_chunk_sizes[3],
                    next_video_chunk_sizes[4],
                    next_video_chunk_sizes[5]])

                # end of video or not
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    done = True
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_count'] = 0
                else:
                    done = False

                # misc info
                info = {'bitrate': VIDEO_BIT_RATE[post_data['lastquality']],
                        'stall time': rebuffer_time}

                action = self.agent.get_action(obs, reward, done, info)

                if done:
                    send_data = 'REFRESH'

                else:
                    send_data = str(action)

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode())

        def do_GET(self):
            print(sys.stderr, 'GOT REQ')
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');".encode())

        def log_message(self, format, *args):
            return

    return Request_Handler


class ABREnv(core.SysEnv):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id=2787486
    """
    def __init__(self):
        # check if the operating system is ubuntu
        if platform != 'linux' and platform != 'linux2':
            raise OSError('Real ABR environment only tested with Ubuntu 16.04.')

        # check/download the video files
        if not os.path.exists(park.__path__[0] + '/envs/abr/video_server/'):
            wget.download(
                'https://www.dropbox.com/s/t1igk37y4qtmtgt/video_server.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/video_server.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')

        # check/download the browser files
        if not os.path.exists(park.__path__[0] + '/envs/abr/abr_browser_dir/'):
            wget.download(
                'https://www.dropbox.com/s/oa0v6s886epkn28/abr_browser_dir.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/abr_browser_dir.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')
            os.system('chmod 777 ' + park.__path__[0] + '/envs/abr/abr_browser_dir/chromedriver')

        # check/download the trace files
        if not os.path.exists(park.__path__[0] + '/envs/abr/cooked_traces/'):
            wget.download(
                'https://www.dropbox.com/s/qw0tmgayh5d6714/cooked_traces.zip?dl=1',
                out=park.__path__[0] + '/envs/abr/')
            with zipfile.ZipFile(
                 park.__path__[0] + '/envs/abr/cooked_traces.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/abr/')

        # check if the manifest file is copied to the right place (last step in setup.py)
        if not os.path.exists('/var/www/html/Manifest.mpd'):
            os.system('python ' + park.__path__[0] + '/envs/abr/setup.py')

        if not os.path.exists(park.__path__[0] + "/envs/abr/local-unix-proxy/target/release/unixskproxy"):
            subprocess.run("git submodule update --init --recursive", shell=True)
            try:
                subprocess.check_call("cd {} && cargo build --release".format(park.__path__[0] + "/envs/abr/local-unix-proxy/"),
                    shell=True)
            except subprocess.CalledProcessError:
                subprocess.check_call("sudo bash {}".format(park.__path__[0] + "/envs/congestion_control/rust-install.sh"), cwd=park.__path__[0] + "/envs/abr", shell=True)
                subprocess.run("cd {} && ~/.cargo/bin/cargo build --release".format(park.__path__[0] + "/envs/abr/local-unix-proxy/"),
                    shell=True)

        # observation and action space
        self.setup_space()

        # load all trace file names
        self.all_traces = os.listdir(park.__path__[0] + '/envs/abr/cooked_traces/')

        # random seed
        self.seed(config.seed)

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0] * 11)
        self.obs_high = np.array([
            10e6, 100, 100, 500, 5, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def run(self, agent, *args, **kwargs):
        # kill all previously running programs
        os.system("ps aux | grep -ie mm-delay | awk '{print $2}' | xargs kill -9")
        os.system("ps aux | grep -ie mm-link | awk '{print $2}' | xargs kill -9")
        os.system("ps aux | grep -ie abr | awk '{print $2}' | xargs kill -9")
        os.system("sudo sysctl -w net.ipv4.ip_forward=1")

        trace_file = self.np_random.choice(self.all_traces)

        # Akshay: use mahimahi base ip instead
        # ip_data = json.loads(urlopen("http://ip.jsontest.com/").read().decode('utf-8'))
        # ip = str(ip_data['ip'])

        # set up agent in the abr server
        input_dict = {'agent': agent,
                  'last_bit_rate': DEFAULT_QUALITY,
                  'last_total_rebuf': 0,
                  'video_chunk_count': 0}

        # remove socket binding
        socket_path = '/tmp/abr_http_socket'
        os.system('rm ' + socket_path)
        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)
        server = UnixHTTPServer(socket_path, handler_class)
        print('Unix socket server starts listening')

        def run_server_forever(server):
            server.serve_forever()

        t = threading.Thread(target=run_server_forever, args=(server,))
        t.start()

        # start real ABR environment
        p = subprocess.Popen('mm-delay 40' +
            ' mm-link ' + park.__path__[0] + '/envs/abr/12mbps ' +
            park.__path__[0] + '/envs/abr/cooked_traces/' + trace_file +
            ' /usr/bin/python3 ' + park.__path__[0] + '/envs/abr/run_video.py ' +
            '320' + ' ' + '0' + ' ' + '1',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True)

        p.wait()
        server.stop = True
        print('Socket server shutdown')
