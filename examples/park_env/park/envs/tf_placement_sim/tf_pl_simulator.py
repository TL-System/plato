from park.envs.tf_placement_sim.tf_sim import Simulator, get_op_costs

class ImportantOpsSimulator(Simulator):

  def __init__(self, mg, op_perf, step_stats, devices):

    cost_d, _ = get_op_costs(step_stats)

    out_d = {}
    for op in op_perf:
      out_d[op.node] = op.op_memory.output_memory

    for dev_stats in step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
                node = node_stats.node_name
                for output in node_stats.output:
                    allocation = output.tensor_description.allocation_description
                    num_bytes = allocation.requested_bytes
                    out_d[node] = [num_bytes]
                    break

    for i, dev in enumerate(devices):
      devices[i] = '/' + dev.split('/')[-1]

    for node in mg.graph_def.node:
      d = node.device
      node.device = '/' + d.split('/')[-1]

    Simulator.__init__(self, mg, cost_d, out_d, devices)

  def simulate(self, pl, sim_mem_usage=False):
    
    for k, v in pl.items():
      pl[k] = self.devices[int(v)]

    r, f = Simulator.simulate(self, pl)

    self.f = f

    start_t = {}
    for node in self.metagraph.graph_def.node:
      n = node.name
      start_t[n] = f[n].start_time

    if sim_mem_usage:

        mem_q = []

        for n, t in start_t.items():

            mem = sum(self.output_dict[n])
            if mem == 0:
                continue

            dev = self.devices.index(f[n].device)

            mem_q.append((t, '+', mem, dev))

            t_out_done = t
            for c in f[n].children:
                t_out_done = max(t_out_done, int(f[c].start_time) + int(f[c].compute_cost) - 1)

            mem_q.append((t_out_done, '-', -mem, dev))

        mem_q.sort()

        mem_utils = [0]* len(self.devices)
        peak_utils = [0]* len(self.devices)

        for (t, _, mem, dev) in mem_q:
            mem_utils[dev] += mem

            if mem_utils[dev] > peak_utils[dev]:
              peak_utils[dev] = mem_utils[dev]

        return r, peak_utils
      
    return r
