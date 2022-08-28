import park
from park.core import Env, Space
from park.envs.multi_dim_index.params import Params as params
from park.envs.multi_dim_index.spaces import ActionSpace, DataObsSpace, QueryObsSpace
from park.envs.multi_dim_index.config import Action, DataObs, QueryObs, Query
from park.envs.multi_dim_index.gen_osm_queries import QueryGen
from park.envs.multi_dim_index.monotonic_rmi2 import MonotonicRMI
import numpy as np
from park.spaces.tuple_space import Tuple
import wget
import os
import random
import stat
import sys
import subprocess
from timeit import default_timer as timer

class MultiDimIndexEnv(Env):
    metadata = {'env.name': 'multi_dim_index'}
    # Rewards are reported as throughput (queries per second)
    reward_range = (0, 1e6)
    action_space = ActionSpace()
    observation_space = Tuple((DataObsSpace, QueryObsSpace))

    def __init__(self):
        datafile = params.DATASET_PATH
        if not os.path.exists(datafile):
            dr = os.path.split(datafile)[0]
            if not os.path.isdir(dr):
                os.makedirs(dr)
            print('Downloading dataset...')
            wget.download(params.DATA_DOWNLOAD_URL, out=datafile)
            # Newline because wget doesn't print it out
            print('')
        binary = params.BINARY_PATH
        if not os.path.exists(binary):
            dr = os.path.split(binary)[0]
            if not os.path.isdir(dr):
                os.makedirs(dr)
            print('Downloading binary...')
            wget.download(params.BINARY_DOWNLOAD_URL, out=binary)
            os.chmod(binary, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print('')

        self.gen_data_summary()
        self.step_count = 0
        self.query_generator = None

    def parse_cmd_output(self, output):
        lines = output.split('\n')
        times = []
        for line in lines:
            if line.startswith('Query'):
                time = int(line.split(':')[1].strip())
                times.append(time)
        return times

    def step(self, action):
        assert self.action_space.contains(action)
        layout_filename = park.__path__[0] + "/envs/multi_dim_index/mdi_layout.dat"
        action.tofile(layout_filename)
        
        print('Generating next query workload...')
        start = timer()
        new_queries = []
        for _ in range(params.QUERIES_PER_STEP):
            q = self.query_generator.random_query()
            new_queries.append(q)
        query_filename = park.__path__[0] + "/envs/multi_dim_index/queries.bin"
        np.array(new_queries).tofile(query_filename)
        end = timer()
        print('Generating query workload took', end-start, 's')

        print('Running range query workload...')
        start = timer()
        cmd = [params.BINARY_PATH, "--dataset=%s" % params.DATASET_PATH,
                "--workload=%s" % query_filename, "--projector=%s" % layout_filename,
                "--folder=%s" % params.DATA_SUMMARY_DIR]
        print(' '.join(cmd))
        outfile = 'cmd_output.txt'
        done = subprocess.run(cmd, stdout=open(outfile, 'w'), stderr=subprocess.STDOUT, encoding='utf-8')
        if done.returncode != 0:
            raise Exception('Query binary did not finish successfully')
        times = []
        times = self.parse_cmd_output(open(outfile).read())
        if len(times) != len(new_queries):
            raise Exception('Results from binary are incomplete')
        end = timer()
        print('Running range query workload took', end-start, 's')

        reward = 1./np.mean(times)
        obs = (DataObs(params.DATASET_PATH), QueryObs(new_queries))
        self.step_count += 1

        # The query times are given as information.
        return obs, reward, self.step_count >= params.STEPS_PER_EPOCH, {"times": times}

    def reset(self):
        self.step_count = 0
        # Restart the query generator with a new random configuration.
        print('Initializing OSM Query generator...')
        start = timer()
        self.query_generator = QueryGen(params.DATASET_PATH)
        end = timer()
        print('Initializing OSM Query generator took', end-start, 's')

    def seed(self, seed=None):
        if seed is not None:
            self.query_generator.seed(seed)
            random.seed(seed+5)

    # Generates a coarse summary of each data dimension, which the indexer uses to divvy up data
    # into columns.
    def gen_data_summary(self):
        data = np.fromfile(params.DATASET_PATH, dtype=np.int64).reshape(-1, params.NDIMS) 
        for ix in range(params.NDIMS):
            filename = f'{params.DATA_SUMMARY_DIR}/dim_{ix}_cdf.dat'
            print(f'Generating CDF for dimension {ix}')
            # Generate a CDF for that dimension
            dim_data = np.sort(data[:, ix]).reshape(-1, 1)
            dim_data_unique, cdf_range_unique = self.unique_cdf(dim_data, mode='bottom') 
            expert_sizes = self.sizes_from_uniques(dim_data_unique, [100, 1000])
            cdf = MonotonicRMI(expert_sizes, last_layer_monotonic=True)
            cdf.fit(dim_data_unique, cdf_range_unique, verbose=False)
            cdf.dump(filename)

    # When there are multiple points with the same value:
    #  - 'middle': the CDF of a point should be the location of the middle point
    #  - 'top': the CDF should be the location of the last of the points with the same value
    #  - 'bottom': the CDF should be the location of the first point
    def unique_cdf(self, xs, mode='middle'):
        uq, inds, counts = np.unique(xs, return_inverse=True, return_counts=True)
        cdf = None
        if mode == 'top':
            cum_counts = np.cumsum(counts).astype(float) / len(xs)
            cdf = cum_counts[inds]
            xs = np.insert(xs, [0], [xs.min()-1], axis=0)
            cdf = np.insert(cdf, 0, 0.0)
        elif mode == 'middle':
            cum_counts = (np.cumsum(counts) - (counts+1)/2).astype(float) / len(xs)
            cdf = cum_counts[inds]
        elif mode == 'bottom':
            cum_counts = (np.cumsum(counts) - counts).astype(float) / len(xs)
            cdf = cum_counts[inds]
            cdf = np.insert(cdf, len(xs), 1.0)
            xs = np.insert(xs, [len(xs)], [xs.max()+1], axis=0)
        return xs, cdf
        
    def sizes_from_uniques(self, data, locator_experts):
        uq = np.unique(data)
        if len(np.unique(uq)) < locator_experts[-1]:
            return [1, int(np.sqrt(len(uq))), len(uq)]
        return locator_experts

