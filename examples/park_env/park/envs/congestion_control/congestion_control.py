from time import time, sleep
import math
import numpy as np
import os
import socket
import subprocess as sh
import sys
import threading
import wget
import zipfile

import park
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.utils.colorful_print import print_red
from park.spaces.box import Box

try:
    import capnp
except:
    sh.run("pip3 install --user pycapnp", shell=True)
    import capnp

capnp.remove_import_hook()
ccp_capnp = capnp.load(park.__path__[0]+"/envs/congestion_control/park/ccp.capnp")

class CcpRlAgentImpl(ccp_capnp.RLAgent.Server):
    def __init__(self, agent):
        self.agent = agent
        self.min_rtt = 0x3fffffff
        self.old_obs = None

    def getReward(self, obs):
        # copa's utility function: log(throughput) - delta * log(delay)
        delta = 0.5
        tput = obs.rout
        delay = obs.rtt - self.min_rtt
        delay = delay if delay > 0 else 0

        logtput = math.log2(tput) if tput > 0 else 0
        logdelay = math.log2(delay) if delay > 0 else 0
        return (logtput - delta * logdelay, (tput, delta, delay))

    def getAction(self, observation, _context, **kwargs):
        obs = [
            observation.bytesAcked,
            observation.bytesMisordered,
            observation.ecnBytes,
            observation.packetsAcked,
            observation.packetsMisordered,
            observation.ecnPackets,
            observation.loss,
            observation.timeout,
            observation.bytesInFlight,
            observation.packetsInFlight,
            observation.bytesPending,
            observation.rtt,
            observation.rin,
            observation.rout,
        ]

        if observation.rtt < self.min_rtt:
            self.min_rtt = observation.rtt

        # get_action(current observation, reward for previous observation, False, components of reward calculation)
        if self.old_obs is not None:
            reward, info = self.getReward(self.old_obs)
        else:
            reward, info = (0, (0, 0, 0))

        try:
            act = self.agent.get_action(obs, reward, False, info)
        except Exception as e:
            print_red("RL Agent failed within get_action: " + str(e))

        self.old_obs = observation

        c, r = act
        action = ccp_capnp.Action.new_message(cwnd=int(c), rate=int(r))
        return action

class ShimAgent(object):
    def __init__(self):
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def get_action(self, *args):
        if self.agent is not None:
            return self.agent.get_action(*args)
        else:
            return (0,0)

global_agent = ShimAgent()

def run_forever(addr, agent):
    server = capnp.TwoPartyServer(addr, bootstrap=CcpRlAgentImpl(agent))
    logger.info("Started RL agent in RPC server thread")
    server.run_forever()

def write_const_mm_trace(outfile, bw):
    with open(outfile, 'w') as f:
        lines = bw // 12
        for _ in range(lines):
            f.write("1\n")

class CongestionControlEnv(core.SysEnv):
    def __init__(self):
        # check if the operating system is ubuntu
        if sys.platform != 'linux' and sys.platform != 'linux2':
            raise OSError('Congetsion control environment only tested on Linux.')

        if os.getuid() == 0:
            raise OSError('Please run as non-root')

        logger.info("Install Dependencies")
        sh.run("sudo apt install -y git build-essential autoconf automake capnproto iperf", stdout=sh.PIPE, stderr=sh.PIPE, shell=True)
        sh.run("sudo add-apt-repository -y ppa:keithw/mahimahi", stdout=sh.PIPE, stderr=sh.PIPE, shell=True)
        sh.run("sudo apt-get -y update", stdout=sh.PIPE, stderr=sh.PIPE, shell=True)
        sh.run("sudo apt-get -y install mahimahi", stdout=sh.PIPE, stderr=sh.PIPE, shell=True)
        sh.run("sudo sysctl -w net.ipv4.ip_forward=1", stdout=sh.PIPE, stderr=sh.PIPE, shell=True)

        sh.run("sudo rm -rf /tmp/park-ccp", shell=True)
        self.setup_ccp_shim()
        self.setup_mahimahi()

        # state_space
        #
        # biggest BDP = 1200 packets = 1.8e6 Bytes
        #
        # bytesAcked         UInt64; at most one BDP
        # bytesMisordered    UInt64; at most one BDP
        # ecnBytes           UInt64; at most one BDP
        # packetsAcked       UInt64; at most one BDP / MSS
        # packetsMisordered  UInt64; at most one BDP / MSS
        # ecnPackets         UInt64; at most one BDP / MSS
        # loss               UInt64; at most one BDP / MSS
        # timeout            Bool;
        # bytesInFlight      UInt64; at most one BDP
        # packetsInFlight    UInt64; at most one BDP / MSS
        # bytesPending       UInt64; ignore
        # rtt                UInt64; [0ms, 300ms]
        # rin                UInt64; [0 Byte/s, 1GByte/s]
        # rout               UInt64; [0 Byte/s, 1GByte/s]
        self.observation_space = Box(
            low=np.array([0] * 14),
            high=np.array([1.8e6, 1.8e6, 1.8e6, 1200, 1200, 1200, 1200, 1, 1.8e6, 1200, 0, 300e3, 1e9, 1e9]),
        )

        # action_space
        # cwnd = [0, 4800 = 4BDP]
        # rate = [0, 2e9 = 2 * max rate]
        self.action_space = Box(low=np.array([0, 0]), high=np.array([4800, 2e9]))

        # kill old shim process
        sh.run("sudo pkill -9 park", shell=True)
        sh.Popen(self.workloadGeneratorKiller, shell=True).wait()
        sleep(1.0)  # pkill has delay

        # start rlagent rpc server that ccp talks to
        logger.info("Starting RPC server thread")
        global global_agent
        t = threading.Thread(target=run_forever, args=("*:4539", global_agent))
        t.daemon = True
        t.start()

        # start ccp shim
        logger.info("Starting CCP shim process")
        cong_env_path = park.__path__[0] + "/envs/congestion_control"
        sh.Popen("sudo " + os.path.join(cong_env_path, "park/target/release/park 2> /dev/null"), shell=True)
        sleep(1.0)  # spawn has delay

        # start workload generator receiver
        sh.Popen(self.workloadGeneratorReceiver, shell=True)

        logger.info("Done with init")

    def run(self, agent):
        logger.info("Setup agent")
        global global_agent
        global_agent.set_agent(agent)

        # Start
        self.reset()

    def reset(self):
        # start Mahimahi
        config_dict = {}

        config_dict["mmdelay"] = "mm-delay"
        config_dict["mmlink"] = "mm-link"
        config_dict["delay"] = int(self.linkDelay)

        config_dict["uplinktrace"] = self.uplinkTraceFile
        config_dict["downlinktrace"] = self.downlinkTraceFile

        config_dict["workloadSender"] = "./sender.sh"

        start_mahimahi_cmd = \
                "%(mmdelay)s %(delay)d %(mmlink)s %(uplinktrace)s %(downlinktrace)s  \
                --uplink-queue=droptail --downlink-queue=droptail --uplink-queue-args=\"packets=2000\" --downlink-queue-args=\"packets=2000\"\
                %(workloadSender)s "% config_dict

        sh.run(start_mahimahi_cmd, shell=True)
        sleep(1.0)

    def setup_mahimahi(self):
        logger.info("Mahimahi setup")

        env_path = park.__path__[0] + "/envs/congestion_control"
        traces_path = os.path.join(env_path, 'traces/')

        if not os.path.exists(traces_path):
            sh.run("mkdir -p {}".format(traces_path), shell=True)
            wget.download(
                'https://www.dropbox.com/s/qw0tmgayh5d6714/cooked_traces.zip?dl=1',
                out=env_path
            )
            with zipfile.ZipFile(env_path + '/cooked_traces.zip', 'r') as zip_f:
                zip_f.extractall(traces_path)

            sh.run("rm -f {}".format(env_path + '/cooked_traces.zip'), shell=True)
            sh.run("cp /usr/share/mahimahi/traces/* {}".format(traces_path), shell=True)

            # const traces
            write_const_mm_trace(os.path.join(traces_path, "const12.mahi"), 12)
            write_const_mm_trace(os.path.join(traces_path, "const24.mahi"), 24)
            write_const_mm_trace(os.path.join(traces_path, "const36.mahi"), 36)
            write_const_mm_trace(os.path.join(traces_path, "const48.mahi"), 48)
            write_const_mm_trace(os.path.join(traces_path, "const60.mahi"), 60)
            write_const_mm_trace(os.path.join(traces_path, "const72.mahi"), 72)
            write_const_mm_trace(os.path.join(traces_path, "const84.mahi"), 84)
            write_const_mm_trace(os.path.join(traces_path, "const96.mahi"), 96)

        # Setup link
        logger.debug(park.param.config)
        self.linkDelay = park.param.config.cc_delay
        self.uplinkTraceFile = os.path.join(traces_path, park.param.config.cc_uplink_trace)
        self.downlinkTraceFile = os.path.join(traces_path, park.param.config.cc_downlink_trace)

        # Setup workload generator
        self.workloadGeneratorSender   = "iperf -c 100.64.0.1 -Z ccp -P 1 -i 2 -t {}".format(park.param.config.cc_duration)
        self.workloadGeneratorReceiver = "iperf -s -w 16m > /dev/null"
        self.workloadGeneratorKiller   = "sudo pkill -9 iperf"

        with open("sender.sh","w") as fout:
            fout.write(self.workloadGeneratorSender+"\n")

        sh.Popen("chmod a+x sender.sh", shell=True).wait()

    def setup_ccp_shim(self):
        cong_env_path = park.__path__[0] + "/envs/congestion_control"

        # ccp-kernel
        if not os.path.exists(cong_env_path + "/ccp-kernel"):
            logger.info("Downloading ccp-kernel")
            sh.run("git clone --recursive https://github.com/ccp-project/ccp-kernel.git {}".format(cong_env_path + "/ccp-kernel"), shell=True)

        try:
            sh.check_call("lsmod | grep ccp", shell=True)
            sh.run("make && sudo ./ccp_kernel_unload && sudo ./ccp_kernel_load ipc=0", cwd=cong_env_path + "/ccp-kernel", shell=True)
        except sh.CalledProcessError:
            logger.info('Loading ccp-kernel')
            sh.run("make && sudo ./ccp_kernel_load ipc=0", cwd=cong_env_path + "/ccp-kernel", shell=True)

        try:
            logger.info("Building ccp shim")
            sh.check_call("cargo build --release", cwd=cong_env_path + "/park", shell=True)
        except sh.CalledProcessError:
            logger.info("Installing rust")
            sh.check_call("sudo bash rust-install.sh", cwd=cong_env_path, shell=True)
            logger.info("Building ccp shim")
            sh.check_call("~/.cargo/bin/cargo build --release", cwd=cong_env_path + "/park", shell=True)
