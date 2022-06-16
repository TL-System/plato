import park 
from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.spaces.box import Box

import numpy as np
from subprocess import Popen
import os
import zmq

from time import time, sleep

from park.envs.aqm.mahimahi_interface import MahimahiInterface

try:
    from park.envs.aqm.ipc_msg_pb2 import IPCMessage, IPCReply
except:
    os.system("protoc -I=" + park.__path__[0] + "/envs/aqm/ --python_out=" + park.__path__[0] + "/envs/aqm/ " + park.__path__[0] + "/envs/aqm/ipc_msg.proto")
    from park.envs.aqm.ipc_msg_pb2 import IPCMessage, IPCReply
class AQMEnv(core.Env):
    """
    TODO: write a description
    """
    def __init__(self):
        
        # Setup Mahimahi

        # TODO: sudo apt-get install apache2-dev
        # sudo apt-get install libssl-dev
        # sudo apt-get install libxcb-xrm-dev 
        # libxcb-present-dev
        # libcairo2-dev
        # libpango1.0-dev
        # sudo apt-get install libzmq3-dev

        # TODO: check os
        mahimahi_path = park.__path__[0]+"/envs/aqm/mahimahi"

        if os.path.exists (mahimahi_path) == False :
            # create folder 
            Popen("mkdir %s" % mahimahi_path, shell=True).wait()
            # get mahimahi
            # Popen("cd %s; git clone https://github.com/songtaohe/mahimahi.git" % (park.__path__[0]+"/envs/aqm/"), shell=True).wait()
            Popen("cd %s; git clone https://github.com/mehrdadkhani/mahimahi-1 mahimahi" % (park.__path__[0]+"/envs/aqm/"), shell=True).wait()
            # Make mahimahi
            # Popen("cd %s; git fetch; git checkout mahimahi_stable2; ./autogen.sh; ./configure; make; sudo make install" % mahimahi_path, shell=True).wait()
            Popen("cd %s; git fetch; git checkout mahimahi_stable2; ./autogen.sh; ./configure; make; sudo make install" % mahimahi_path, shell=True).wait()

        # compile protoc
        # protoc -I=./park/envs/aqm/ --cpp_out=./park/envs/aqm/mahimahi/src/packet ./park/envs/aqm/ipc_msg.proto

        # TODO: disable pie

        #self.mm_delay = mahimahi_path + "/src/frontend/mm-delay"
        #self.mm_link = mahimahi_path + "/src/frontend/mm-link"
        
        self.mm_delay = "mm-delay"
        self.mm_link = "mm-link"
        
        # Setup link 

        self.linkDelay = config.aqm_link_delay
        self.uplinkTraceFile = config.aqm_uplink_trace
        self.downlinkTraceFile = config.aqm_downlink_trace


        # Setup workload generator

        self.workloadGeneratorSender   = "iperf -c 100.64.0.1 -P 1 -i 2 -t 10000"
        self.workloadGeneratorReceiver = "iperf -s -w 16m"
        self.workloadGeneratorKiller   = "pkill -9 iperf"

        with open("sender.sh","w") as fout:
            fout.write(self.workloadGeneratorSender+"\n")

        Popen("chmod a+x sender.sh", shell=True).wait()


        # Setup RL
        self.aqm_step_interval = config.aqm_step_interval
        self.aqm_step_num = config.aqm_step_num 

        self.state_space = Box(low=np.array([0, 0, 0]), high=np.array([1e4, 1e4, 1e3]))
        self.action_space = Box(low=np.array([0]), high=np.array([1]))

        self.step_counter = 0

        # Start 
        # self.reset()

        self.last_action_ts = None

    def _observe(self):
        ret = self.mahimahi.GetState()
        obs = np.array([ret['enqueued_packet'], ret['dequeued_packet'], ret['average_queueing_delay']])
        reward = -ret["average_queueing_delay"]
        info = {'message': ret['info']}
        return obs, reward, info

    def render(self):
        # TODO: depends on a switch (in __init__), visualize the mahimahi console
        pass


    def step(self, action):
        
        assert self.action_space.contains(action)

        self.step_counter += 1


        if self.last_action_ts is None:
            t_sleep = self.aqm_step_interval/1000.0
            sleep(t_sleep)
        else:
            t0 = time()
            t_sleep = self.last_action_ts + self.aqm_step_interval/1000.0 - t0

            if t_sleep > 0 :
                sleep(t_sleep)

        self.mahimahi.SetDropRate(action[0])
        self.last_action_ts = time()

        obs, reward, info = self._observe()
        assert self.state_space.contains(obs)
        info['wall_time_elapsed'] = t_sleep

        done = False
        if self.step_counter >= self.aqm_step_num:
            done = True

        return obs, reward, done, info
        

    def reset(self):
        # kill Mahimahi and workload generator
        Popen("pkill mm-delay", shell=True).wait()
        Popen(self.workloadGeneratorKiller, shell=True).wait()
        sleep(1.0)  # pkill has delay

        # start workload generator receiver 
        Popen(self.workloadGeneratorReceiver, shell=True)

        # start Mahimahi
        config_dict = {}
        config_dict["mmdelay"] = self.mm_delay
        config_dict["mmlink"] = self.mm_link
        config_dict["delay"] = int(self.linkDelay)
        config_dict["uplinktrace"] = self.uplinkTraceFile
        config_dict["downlinktrace"] = self.downlinkTraceFile
        config_dict["workloadSender"] = "./sender.sh"

        start_mahimahi_cmd = \
        "%(mmdelay)s %(delay)d %(mmlink)s %(uplinktrace)s %(downlinktrace)s  \
        --meter-uplink --meter-uplink-delay --uplink-queue=pie --downlink-queue=infinite --uplink-queue-args=\"packets=2000, qdelay_ref=20, max_burst=1\" \
        %(workloadSender)s "% config_dict

        Popen(start_mahimahi_cmd, shell=True)
        sleep(1.0)  # mahimahi start delay

        # Connect to Mahimahi
        self.mahimahi = MahimahiInterface()
        self.mahimahi.ConnectToMahimahi()

        # Gain control
        #self.mahimahi.SetRLState(1)

        ## get obs 
        #obs, _, _ = self._observe()
        #self.step_counter = 0
        return True
        #return obs 
#    def reset(self):
#        # kill Mahimahi and workload generator
#
#        Popen("pkill mm-delay", shell=True).wait()
#        Popen(self.workloadGeneratorKiller, shell=True).wait()
#
#        sleep(1.0)  # pkill has delay
#
#        # start workload generator receiver 
#        Popen(self.workloadGeneratorReceiver, shell=True)
#
#
#        # start Mahimahi
#        config_dict = {}
#
#        config_dict["mmdelay"] = self.mm_delay
#        config_dict["mmlink"] = self.mm_link
#        config_dict["delay"] = int(self.linkDelay)
#
#        config_dict["uplinktrace"] = self.uplinkTraceFile
#        config_dict["downlinktrace"] = self.downlinkTraceFile
#
#        config_dict["workloadSender"] = "./sender.sh"
#
#        start_mahimahi_cmd = \
#        "%(mmdelay)s %(delay)d %(mmlink)s %(uplinktrace)s %(downlinktrace)s  \
#        --meter-uplink --meter-uplink-delay --uplink-queue=pie --downlink-queue=infinite --uplink-queue-args=\"packets=2000, qdelay_ref=20, max_burst=1\" \
#        %(workloadSender)s "% config_dict
#
#        Popen(start_mahimahi_cmd, shell=True)
#
#        sleep(1.0)  # mahimahi start delay
#
#        # Connect to Mahimahi
#        self.mahimahi = MahimahiInterface()
#
#        self.mahimahi.ConnectToMahimahi()
#
#
#        # Gain control
#        self.mahimahi.SetRLState(1)
#
#        # get obs 
#        obs, _, _ = self._observe()
#        self.step_counter = 0
#
#        return obs 

    def run(self, agent):
        # set up ipc communication
        context = zmq.Context(1)
        socket = context.socket(zmq.REP)
        ipc_msg = IPCMessage()
        ipc_reply = IPCReply()

        os.system('rm /tmp/aqm_cpp_python_ipc')
        socket.bind("ipc:///tmp/aqm_cpp_python_ipc")
        self.reset() 
        
        last_eqc = 0 
        last_dqc = 0 
        last_qdelay = 0 

        eqc = 0 
        dqc = 0 
        qdelay = 0

        reg = 0

        while True:
            #print('inside')
            msg = socket.recv()
            ipc_msg.ParseFromString(msg)
            #print('got message', ipc_msg.msg)

            if ipc_msg.msg == 'get_prob':
                try:
                    eqc = ipc_msg.eqc 
                    dqc = ipc_msg.dqc 
                    qdelay = ipc_msg.qdelay 
                except:
                    print("error reading ipc msg")

                avg_queue_delay = (qdelay - last_qdelay)/float(dqc-last_dqc + 0.01)
                eq_counter = eqc - last_eqc 
                dq_counter = dqc - last_dqc 

                last_qdelay = qdelay
                last_eqc = eqc 
                last_dqc = dqc 

                #print(avg_queue_delay, eq_counter, dq_counter)

                reward = -abs(avg_queue_delay-20) + reg 

                prob = 0.2*agent.get_action(np.array([avg_queue_delay, eq_counter, dq_counter]), reward)

                prob = prob[0][0]
                #print(prob)

                if prob > 0.5 :
                    reg = -10.0 * (prob-0.5)

                if prob < 0.0:
                    reg = -10.0 * (0.0-prob)

                



                ipc_reply.prob = prob#agent(ipc_msg.state) 
                ipc_reply.msg = 'set_prob'            
                socket.send(ipc_reply.SerializeToString(),0)

                sleep(0.1)

    def seed(self, seed=None):
        # no controllable randomness
        pass

    def clean(self):
        Popen("pkill mm-delay", shell=True).wait()
        Popen(self.workloadGeneratorKiller, shell=True).wait()

        Popen("rm mahimahi_pipe1 mahimahi_pipe2 sender.sh", shell=True).wait()

        sleep(1.0)
