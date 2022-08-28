import numpy as np

from park import core, spaces, logger
from park.param import config
from park.utils import seeding


class SimpleQueueEnv(core.Env):
    """
    Balance the load among n (default 5) homogeneous servers to
    maximize the priority of job being admitted. At each time step,
    each server has a probability p to be free up. Example from
    Sutton RL book example 10.2.

    * STATE *
        Priority of incoming job, number of free servers.

    * ACTIONS *
        Admit the job or not {0, 1}.

    * REWARD *
        Job priority.
    
    * REFERENCE *
        Example 10.2, Chapter 10.3
        Sutton, Richard S., and Andrew G. Barto.
        Reinforcement learning: An introduction. (Edition 2)
        MIT press, 2018.
    """
    def __init__(self):
        # observation and action space
        self.setup_space()
        # random seed
        self.seed(config.seed)


    def observe(self):
        obs_arr = np.array([sum(self.servers), self.incoming_job])
        assert self.observation_space.contains(obs_arr)
        return obs_arr

    def reset(self):
        # reset n servers
        self.servers = [0] * config.sq_num_servers
        # incoming job priority
        self.incoming_job = self.np_random.randint(1, 6)
        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.obs_low = np.array([0, 1])
        self.obs_high = np.array([config.sq_num_servers, 5])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def step(self, action):

        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        if action == 0:
            # evict the job
            reward = 0

        else:
            # admit the job
            admitted = False
            for i in range(len(self.servers)):
                if self.servers[i] == 0:
                    self.servers[i] = 1
                    admitted = True
                    break
            if admitted:
                reward = self.incoming_job
            else:
                reward = 0

        # generate the next incoming job
        self.incoming_job = self.np_random.randint(1, 6)

        # each server has some probability to free up
        for i in range(len(self.servers)):
            if self.servers[i] and self.np_random.rand() < config.sq_free_up_prob:
                self.servers[i] = 0

        done = False

        return self.observe(), reward, done, {}
