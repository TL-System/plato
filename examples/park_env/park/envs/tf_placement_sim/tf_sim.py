import math
import queue
from copy import deepcopy
from collections import defaultdict
import heapq

def get_op_costs(step_stats):
  d = {}
  cost_d = {}

  for dev_stat in step_stats.dev_stats:
    # https://github.com/tensorflow/tensorflow/blob/4595f1cff635ce024e875f0f3d480172731b0b22/tensorflow/core/profiler/internal/tfprof_node.cc
    if 'all' in dev_stat.device: #or 'CPU' in dev_stat.device:
    # if 'cpu' not in dev_stat.device.lower():
        for node_stat in dev_stat.node_stats:
            n = node_stat.node_name.split(':')[0]
            if n not in d:
              d[n] = [node_stat.all_start_micros, node_stat.all_end_rel_micros - \
                node_stat.op_start_rel_micros]
            else:
              d[n][1] += node_stat.all_end_rel_micros - node_stat.op_start_rel_micros

            cost_d[n] = d[n][1]

  return cost_d, d

class SimQueue(object):
  def __init__(self):
    self.queue = []

  def put(self, x):
    heapq.heappush(self.queue, x)

  def get(self):
    return heapq.heappop(self.queue)

  def empty(self):
    return len(self.queue) == 0


class Simulator(object):
  """ Simulator class """
  
  class Node(object):
    """ 
      Node class
      Used to store the attributes of node in graph
    """
    pass

  # default params optimized for p2.8xlarge instance 
  # on AWS EC2 service
  default_params = {
      # wait for delta1 after an op is done
      "delta1" : 5.7, # us
      # constant time overhead for transfer
      "delta2" : 25, # us
      "init_offset" : 0, # us
      "transfer_speed" : 7600 # bytes/us
  }

  def __init__(self, metagraph, cost_dict, output_dict, devices):
    """
      Init for Simulator class
      Args:
        metagraph: metagraph to simulate
        cost_dict: contains run_time of nodes in microseconds
        output_dict: contains output sizes of nodes
        devices: list of device names
        params: dictionary of parameters to use in simulator
          some parameters can be missing
    """
    self.metagraph = metagraph
    self.cost_d = self.cost_dict = defaultdict(int, cost_dict)
    self.out_d = self.output_dict = defaultdict(list, output_dict)
    self.bus = "/bus"
    self.devices = devices
    self.params = self.default_params
    self.node_dict = self.get_attributes()

    # Make a parent_map : node_name -> bool map of parents
    self.parent_map = defaultdict(dict)
    for k,v in self.node_dict.items():
      for p in v.parents:
        self.parent_map[k][p] = True

  def get_attributes(self):
    """ 
      Creates the node_dict. Node contains the following
      Attributes
        op_name: name of op
        device: device of the node
        compute_cost: run time in ns
        output_memory: list of output sizes
        parents: set of parents
        children: dict from node_name to output_index
    """
    # Create a dict from node_name -> Node
    f = dict()

    # Set default values
    for node in self.metagraph.graph_def.node:
      f[node.name] = self.Node()
      f[node.name].op_name = node.op
      f[node.name].device = node.device
      f[node.name].compute_cost = self.cost_dict[node.name]
      f[node.name].output_memory = self.output_dict[node.name]
      f[node.name].parents = set()
      f[node.name].children = dict()
    
    # iterate through all the nodes of the graph
    for node in self.metagraph.graph_def.node:
      # If neither CPU or GPU, then put on CPU
      for i in node.input:
        i = i[1:] if i[0] is '^' else i
        i = (i + ":0") if ":" not in i else i
        i = i.split(":")
        # Set parents and children
        parent, out_idx = i[0], int(i[1])
        while out_idx >= len(f[parent].output_memory):
          f[parent].output_memory.append(0)
        f[node.name].parents.add(parent)
        f[parent].children[node.name] = out_idx

    return f

  def simulate(self, device_dict=dict()):

    """
      Run the simulation
      Args:
        device_dict: Contains mapping from device_name to device
          May be incomplete. Default mapping used if incomplete
      Return:
        tuple of (run_time, node_dict)
    """
    

    i, run_time = 0, 0
    # q has event (op/tranfer/remove_dependency) done events
    q = SimQueue()
    f = self.node_dict
    all_dev = self.devices + [self.bus + dev for dev in self.devices]
    # device_in_queue is basically is_device_busy map right now map
    device_in_queue = dict((dev, False) for dev in all_dev)
    # device_queue holds the currently runnable nodes waiting for device to get free
    device_queue = dict((dev, SimQueue()) for dev in all_dev)

    # Reset parent_map
    for k,v in self.parent_map.items():
      for p in v.keys():
        v[p] = True

    def is_scheduleable(n):
      for v in self.parent_map[n].values():
        if v: return False
      return True

    # get the device for node k
    def get_dev(k):
      if k in device_dict:
        return device_dict[k]
      else:
        print('not in device_dict ', k)
        raise Exception('device not assigned for op %s' % k)
        return self.node_dict[k].device
    
    def add_to_dev_queue(t, op, dev, element):
      nonlocal i
      i += 1
      device_queue[dev].put(((t, i), element))
      if not device_in_queue[dev]:
        q.put((t, op, dev))
        device_in_queue[dev] = True

    # Runs the next job on device
    def run_dev(t, dev):
      p, node_name = device_queue[dev].get()
      assert p[0] <= t, "Priority after exec time, p=%d, t=%d" % (p[0], t)
      node = self.node_dict[node_name]
      # Compute start and end times

      # f[node_name] = self.Node()
      f[node_name].start_time = t
      f[node_name].end_time = t + node.compute_cost
      f[node_name].device = dev

      # Schedule when device is free again
      delta = 0 if node.compute_cost is 0 else self.params["delta1"]
      q.put((f[node_name].end_time + delta, "run_dev", dev))
      
      # Find which all output indices require bus
      require_bus = defaultdict(list) # output_index to list of children
      for c, o in node.children.items():
        if dev == get_dev(c):
          q.put((f[node_name].end_time, "remove_dependency", (node_name, c)))
        else:
          require_bus[o].append(c)
      
      # Schedule transfer on bus
      for o, c_list in require_bus.items():
        delay = node.output_memory[o] / self.params["transfer_speed"]
        add_to_dev_queue(f[node_name].end_time, "run_bus", self.bus + dev, (node_name, delay, require_bus[o]))

    # Run bus
    def run_bus(t, dev):
      p, (node_name, delay, child_list) = device_queue[dev].get()
      # If bus is scheduled to run later, run later
      if p[0] > t:
        device_queue[dev].put((p, (node_name, delay, child_list)))
        q.put((p[0], "run_bus", dev))
        return
      for c in child_list:
        q.put((t + delay, "remove_dependency", (node_name, c)))
      q.put((t + delay + self.params["delta2"], "run_bus", dev))

    # Removes dependency of parent from child
    def remove_dependency(t, parent_name, child_name):
      self.parent_map[child_name][parent_name] = False
      # Schedule child if no more dependencies
      if is_scheduleable(child_name):
        add_to_dev_queue(t, "run_dev", get_dev(child_name), child_name)

    # Insert all runnable ops to device_queue
    for name, node in self.node_dict.items():
      if not node.parents:
        add_to_dev_queue(self.params["init_offset"], "run_dev", get_dev(name), name)

    # Main loop
    while not q.empty():
      t, op, dev = q.get()
      run_time = max(run_time, t)
      if (op is "run_bus" or op is "run_dev") and device_queue[dev].empty():
        device_in_queue[dev] = False
        continue
      elif op is "run_bus":
          run_bus(t, dev)
      elif op is "remove_dependency":
        p_name, c_name = dev
        remove_dependency(t, p_name, c_name)
      elif op is "run_dev":
          run_dev(t, dev)

    return run_time, f
