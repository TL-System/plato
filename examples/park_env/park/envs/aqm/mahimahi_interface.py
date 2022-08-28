import socket
import sys
import numpy as np
from time import sleep, time
from numpy import uint32
from numpy import int32
from numpy import uint64
from numpy import int64

from subprocess import Popen
import os 

class MahimahiInterface():
	def __init__(self):
		self.enLog = 1
		self.enRL = 0
		self.RLTargetQlen = 40
		self.RLDrop = 0.1
		
		#self.last_id = 0
		#self.last_sum = 0
		#self.last_abs = 0
		#self.last_square = 0
		#self.last_qdelay = 0
		self.last_dqbyte = 0
		self.last_eqbyte = 0
		self.last_dqpkg = 0
		self.last_eqpkg = 0 
		self.last_qdelay = 0

		#self.last_qempty_time = 0
		#self.last_acc_qdelay = 0
		self.old_dp = 0
		self.action_delay = 0
		self.action_ts = 0

		#Popen("mkfifo mahimahi_pipe1", shell=True).wait()
		#Popen("mkfifo mahimahi_pipe2", shell=True).wait()

		

	def __WriteProc(self):
		cmd = "W "+str(self.enLog) + " " + str(self.enRL)
		cmd = cmd + " " + str(self.RLTargetQlen)
		dp = self.RLDrop
		cmd = cmd + " " + str(dp)
		#print(cmd)
		#Popen("sudo echo '%s' > /proc/rlint" % cmd,shell=True).wait()
		#self.sk.send(cmd)

		pipe = open("mahimahi_pipe1", "w")

		pipe.write(cmd)
		sleep(0.01) # TODO check named pipe sync issue
		pipe.close()

 

	def ConnectToMahimahi(self,ip='100.64.0.3',port=4999):
		return 

		self.TCP_IP = ip
		self.TCP_PORT = port
		print(ip, " ", port)
		self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sk.connect((ip, port))
		self.sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)						
		pass

	def SetLogState(self, state):
		self.enLog = state
		self.__WriteProc()
		pass

	def SetRLState(self, state):
		self.enRL = state
		self.__WriteProc()
		pass

	def SetTargetQlen(self, qlen):
		self.RLTargetQlen = qlen
		self.__WriteProc()
		pass

	def SetDropRate(self, dropRate):
		dropRate = np.minimum(dropRate, 1.0)
		dropRate = np.maximum(dropRate, 0.0)
		self.RLDrop = dropRate # * 0.25 + self.old_dp*0.75
		self.old_dp = self.RLDrop
		self.__WriteProc()
		self.action_delay = time() - self.action_ts
		#print self.action_delay

		pass

	def GetState(self, dump = True):
		## Read State From Mahimahi
		## State[0] Number of packets enqueued since last query 
		## State[1] Reserved
		## State[2] Reserved
		## State[3] Reserved
		## State[4] Reserved
		## State[5] Number of bytes dequeued since last query
		## State[6] Number of packets dequeued since last query
		## State[7] Total Queue Delay of the dequeued packet 
		## State[8] Current drop rate
		
		self.action_ts = time()

		#self.sk.send("R")
		pipe = open("mahimahi_pipe1", "w")
		pipe.write("R")
		sleep(0.01) # TODO Check named pipe sync issue
		pipe.close()

		
		t0 = time()
		pipe = os.open("mahimahi_pipe2", os.O_RDONLY);
		info = os.read(pipe,256);
		os.close(pipe);

		#info = self.sk.recv(256)
		
		info = info.split()

		eqpkg = uint64(info[0]) - self.last_eqpkg
		dqpkg = uint64(info[6]) - self.last_dqpkg

		qDelaySum = uint64(info[7]) - self.last_qdelay
		
		self.last_eqpkg = uint64(info[0])
		self.last_dqpkg = uint64(info[6])
		self.last_qdelay = uint64(info[7])

		qDelayAvg = 0.0

		if dqpkg > 0 :
			qDelayAvg = qDelaySum / float(dqpkg)

		_info = None 
		if dump == True:
			_info = "Enq %4d packet(s), Deq %4d packet(s),  Avg_Q_Len %.2f ms  DropRate %.3f" % (eqpkg, dqpkg, qDelayAvg, self.RLDrop);
			print(_info) 


		ret = {}

		ret['enqueued_packet'] = eqpkg 
		ret['dequeued_packet'] = dqpkg
		ret['average_queueing_delay'] = qDelayAvg
		ret['info'] = _info
		return ret 
