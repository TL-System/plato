import os
import numpy as np
import math
from itertools import permutations
import wget
import pickle
import networkx as nx

import park
from park import core, spaces, logger
from park.utils.misc import create_folder_if_not_exists
from park.spaces import Tuple, Box, Discrete, Graph, Null
from park.param import config
from park.utils import seeding
from park.utils.directed_graph import DirectedGraph
from park.envs.tf_placement_sim.tf_pl_simulator import ImportantOpsSimulator

dropbox_links = {
    'inception': 'https://www.dropbox.com/s/1r5n4e2g3hulsge/inception.pkl?dl=1',
    'nasnet': 'https://www.dropbox.com/s/ufm72htk1zeuccm/nasnet.pkl?dl=1',
    'nmt': 'https://www.dropbox.com/s/9rsmmv6pm11h3i8/nmt-attention-seq-30.pkl?dl=1',
}

pkl_names = {
    'inception': 'inception.pkl',
    'nasnet': 'nasnet.pkl',
    'nmt': 'nmt-attention-seq-30.pkl',
}

class TFPlacementSimEnv(core.Env):
    """
    Assign a placement to each operation group of a
    computational graph of deep-learning models.
    The goal is to minimize runtime of the computational graph. 

    * STATE *
        Directed Graph with node feature being a list of the following:
            (1) Cost: Op group execution time
            (2) Mem: Op group's memory requirement when running
            (3) Curr Placement: device id of the node based on its 
            current placement in the episode
            (4) is_curr_node: Is this the node that is currently being placed
        

    * ACTIONS *
        [0, 1, ..., n-1] where n is the number of devices. The index
        corresponding to the device id.

    * REWARD *
        Improvement in the runtime of the placement because of the current action
    
    * REFERENCE *
        https://arxiv.org/pdf/1706.04972.pdf
    """
    def __init__(self):
        # observation and action space
        self.setup_env()
        self.setup_space()
        # random seed
        self.seed(config.seed)

    def possibly_download_pkl_file(self):
        graph_dir = park.__path__[0] + '/envs/tf_placement_sim/graphs/'
        trace_file = graph_dir + '/' + pkl_names[config.pl_graph]

        create_folder_if_not_exists(graph_dir)

        if not os.path.exists(trace_file):
            wget.download(dropbox_links[config.pl_graph],
                out=graph_dir)

        return trace_file

    def setup_env(self):
        device_names = ['/device:GPU:%d' % i for i in range(config.pl_n_devs)]
        gpu_devs = filter(lambda dev: 'GPU' in dev, device_names)
        gpu_devs = list(sorted(gpu_devs))

        if config.pl_graph not in pkl_names:
            raise Exception('Requesting for model "%s" which doesnot exist in repo.\n\
                                     Please choose from one of the following %s' % \
                                     (config.pl_graph, ' '.join(pkl_repo.keys())))

        pickled_inp_file = self.possibly_download_pkl_file()
        with open(pickled_inp_file, 'rb') as f:
            j = pickle.load(f)
            mg, G, ungroup_map = j['optim_mg'], j['G'], j['ungrouped_mapping']
            op_perf, step_stats = j['op_perf'], j['step_stats']

        self.mg = mg
        self.ungroup_map = ungroup_map
        self.n_devs = config.pl_n_devs
        self.gpu_devs = gpu_devs
        self.devs = self.gpu_devs
        self.device_names = device_names
        self.G = G

        self.sim = ImportantOpsSimulator(mg, op_perf, step_stats, device_names)
        self.node_order = list(nx.topological_sort(G))
        self.cost_d = self.sim.cost_d
        self.out_d = {k: sum(v) for k, v in self.sim.out_d.items()}

    def reset(self):
        node_features = {}
        edge_features = {}
        cur_pl = {}
        for node in self.G.nodes():
            # checkout step function for this order as well
            node_features[node] = [self.cost_d[node],\
                                   self.out_d[node],\
                                   0,\
                                   0]
            cur_pl[node] = node_features[node][2]
            for neigh in self.G.neighbors(node):
                # dummy edge feature for now
                # TODO: Allow for no edge feature possibility
                edge_features[(node, neigh)] = -1

        node_features[self.node_order[0]][-1] = 1

        self.s = DirectedGraph(node_features, edge_features)
        self.cur_node_idx = 0
        self.cur_pl = cur_pl
        self.prev_rt = self.get_rt(self.cur_pl)

        return self.s


    def setup_space(self):
        # cost (e.g., execution delay estimation in micro-seconds),
        # mem (e.g., op group memory requirements on GPU in bytes),
        # current placement(e.g., GPU 1),
        # one-hot-bit (i.e., currently working on this node)

        node_space = Box(low=0, high=10 * (1e9), shape=(len(self.G), 4), dtype=np.float32)
        dummy_edge_space = Box(low=-1, high=-1, shape=(self.G.number_of_edges(),), dtype=np.int8)
        self.observation_space = Graph(node_space, dummy_edge_space)
        self.action_space = Discrete(self.n_devs)

    def ungroup_pl(self, pl):
        ungroup_map = self.ungroup_map
        ungrouped_pl = {}

        for op in self.mg.graph_def.node:
            name = op.name
            grp_ctr = ungroup_map[name]
            ungrouped_pl[name] = pl[grp_ctr] 

        return ungrouped_pl

    # takes op-group placement and 
    # returns runtime of the placement in seconds
    def get_rt(self, pl):
        pl = self.ungroup_pl(pl)
        rt = self.sim.simulate(pl)
        return rt / 1e6


    def step(self, action):
        assert self.action_space.contains(action)

        action = int(action)
        node_order = self.node_order
        cur_node_idx = self.cur_node_idx
        cur_node = node_order[cur_node_idx]
        next_node = node_order[cur_node_idx + 1]

        self.cur_pl[cur_node] = action
        rt = self.get_rt(self.cur_pl)
        reward = rt - self.prev_rt

        self.s.update_nodes({cur_node:\
                            [self.cost_d[cur_node],\
                            self.out_d[cur_node],\
                            int(action),\
                            0], \

                            next_node:\
                                [self.cost_d[next_node],\
                                self.out_d[next_node],\
                                self.cur_pl[next_node],\
                                1]
                        })

        self.cur_node_idx += 1
        self.prev_rt = rt
        if 1 + self.cur_node_idx == len(self.node_order):
            done = True
        else:
            done = False

        assert self.observation_space.contains(self.s)

        return self.s, reward, done, {}

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)
