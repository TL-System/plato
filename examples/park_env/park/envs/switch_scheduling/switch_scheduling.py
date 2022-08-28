import numpy as np
import math
from itertools import permutations

from park import core, spaces, logger
from park.param import config
from park.utils import seeding


class SwitchEnv(core.Env):
    """
    Schedule packet from input switch queues to output ports.
    The goal is to minimize packet delay. The challenge centers
    around the exponentially large action space (permutation of
    all possible mappings).

    * STATE *
        A matrix of current queue occupancy. The (i, j) element in
        the matrix indicates the queue length (number of backlogged
        packets) in the i-th input queue connecting to j-th output
        port.

    * ACTIONS *
        [0, 1, ..., n!-1] where n is the number of ports. The index
        corresponding to the permutation of mapping. The permuation
        is generated using itertools.permutations.
        For example, n=3, ports={0, 1, 2}, itertools.permutations([
        0, 1, 2]) = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0)]. Action 1 corresponds to (0, 2, 1),
        i.e., maps packet from input queue 0 to output port 0,
        input 1 to output 2 and input 2 to output 1.

    * REWARD *
        Negative number of packets remaining in the queue after the 
        action. So sum of reward indicates total packet delay.
    
    * REFERENCE *
        Chapter 4
        Communication networks: an optimization, control, and stochastic networks perspective
        R Srikant and L Ying
    """
    def __init__(self):
        # observation and action space
        self.setup_space()
        # random seed
        self.seed(config.seed)

    def reset(self):
        # generate a bistochastic matrix for traffic
        r = self.np_random.uniform(0, 1, (config.ss_num_ports, config.ss_num_ports))
        # project into bistochastic space
        while True:
            s_c = r.sum(axis=1, keepdims=True)
            r /= s_c
            s_r = r.sum(axis=0)
            r /= s_r

            if (np.abs(s_r - 1) < 1e-3).all() and (np.abs(s_c - 1) < 1e-3).all():
                break
        self.bistochastic_mat = r
        
        # generate all actions
        self.all_mappings = list(permutations(range(config.ss_num_ports)))

        # generate initial traffic
        self.queue_occupancy = self.sample_from_bistochastic_matrix()

        return self.queue_occupancy

    def sample_from_bistochastic_matrix(self):
        incoming_traffic = self.np_random.binomial(1, config.ss_load * self.bistochastic_mat)
        return incoming_traffic

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        self.observation_space = spaces.Box(low=0, high=config.ss_state_max_queue,
            shape=(config.ss_num_ports, config.ss_num_ports))
        self.action_space = spaces.Discrete(math.factorial(config.ss_num_ports))

    def step(self, action):
        # action is a permutation of identity matrix
        assert self.action_space.contains(action)

        # action corresponding to one of the permutation
        port_mapping = self.all_mappings[action]

        # assign packet to the output port
        for (i, p) in enumerate(port_mapping):
            self.queue_occupancy[i, p] = max(self.queue_occupancy[i, p] - 1, 0)

        # reamining total queue length
        reward = - np.sum(self.queue_occupancy)
        # never ending environment
        done = False

        # sample new traffic
        incoming_traffic = self.sample_from_bistochastic_matrix()
        self.queue_occupancy += incoming_traffic

        # cap the observation
        if np.any(self.queue_occupancy > config.ss_state_max_queue):
            obs_queue = np.minimum(
                self.queue_occupancy, config.ss_state_max_queue)
            logger.warn('Queue occupancy is clipped since it exceeds max queue value ' +
                        str(config.ss_state_max_queue))
        else:
            obs_queue = self.queue_occupancy


        # current queue occupancy in the observation space
        assert self.observation_space.contains(obs_queue)

        return obs_queue, reward, done, {}
