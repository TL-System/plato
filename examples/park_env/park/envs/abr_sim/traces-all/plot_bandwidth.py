import numpy as np
import matplotlib.pyplot as plt


TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100
LINK_FILE = './3bandwidth_generated_changing.log'


bandwidth_all, time_all = [], []
with open(LINK_FILE, 'r') as f:
	for line in f:
		throughput = int(float(line.split()[1]))
		bandwidth_all.append(throughput)
		time = int(float(line.split()[0]))
		time_all.append(time)

bandwidth_all = np.array(bandwidth_all)
bandwidth_all = bandwidth_all * BITS_IN_BYTE / MBITS_IN_BITS
time_all = np.array(time_all)

plt.plot(time_all, bandwidth_all)
plt.xlabel('Time (second)')
plt.ylabel('Throughput (Mbit/sec)')
plt.show()
