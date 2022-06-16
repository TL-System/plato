import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import heapq

from park import core, spaces, logger
from park.param import config
from park.utils import seeding
from park.envs.cache.trace_loader import load_traces

accept = 1
reject = 0


class TraceSrc(object):
    '''
    Tracesrc is the Trace Loader

    @param trace: The file name of the trace file
    @param cache_size: The fixed size of the whole cache
    @param load_trace: The list of trace data. The item could be gotten by using load_trace.iloc[self.idx]
    @param n_request: length of the trace
    @param min_values, max values: Used for the restricted of the value space
    @param req: The obj_time of the object
    '''

    def __init__(self, trace, cache_size):
        self.trace = trace
        self.cache_size = cache_size
        self.load_trace = load_traces(self.trace, self.cache_size, 0)
        self.n_request = len(self.load_trace)
        self.cache_size = cache_size
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.req = 0

    def reset(self, random):
        if self.trace == 'test':
            self.load_trace = load_traces(self.trace, self.cache_size, random)
        self.n_request = len(self.load_trace)
        self.min_values = np.asarray([1, 0, 0])
        self.max_values = np.asarray([self.cache_size, self.cache_size, max(self.load_trace[0])])
        self.req = 0

    def step(self):
        #  Obs is: (obj_time, obj_id, obj_size)
        #  print("req id in trace step:", self.req)
        obs = self.load_trace.iloc[self.req].values
        self.req += 1
        done = self.req >= self.n_request
        return obs, done

    def next(self):
        obs = self.load_trace.iloc[self.req].values
        done = (self.req + 1) >= self.n_request
        return obs, done

class CacheSim(object):
    def __init__(self, cache_size, policy, action_space, state_space):
        # invariant
        '''
        This is the simulater for the cache.
        @param cache_size
        @param policy: Not implement yet. Maybe we should instead put this part in the action
        @param action_space: The restriction for action_space. For the cache admission agent, it is [0, 1]: 0 is for reject and 1 is for admit
        @param req: It is the idx for the object requiration
        @param non_cache: It is the list for those requiration that aren't cached, have obj_id and last req time
        @param cache: It is the list for those requiration that are cached, have obj id and last req time
        @param count_ohr: ohr is (sigma hit) / req
        @param count_bhr: ohr is (sigma object_size * hit) / sigma object_size
        @param size_all: size_all is sigma object_size
        '''

        self.cache_size = cache_size
        self.policy = policy
        self.action_space = action_space
        self.observation_space = state_space
        self.req = 0
        self.non_cache = defaultdict(list)
        self.cache = defaultdict(list)  # requested items with caching
        self.cache_pq = []
        self.cache_remain = self.cache_size
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0

    def reset(self):
        self.req = 0
        self.non_cache = defaultdict(list)
        self.cache = defaultdict(list)
        self.cache_pq = []
        self.cache_remain = self.cache_size
        self.count_ohr = 0
        self.count_bhr = 0
        self.size_all = 0

    def step(self, action, obj):
        req = self.req
        # print(self.req)
        cache_size_online_remain = self.cache_remain
        discard_obj_if_admit = []
        obj_time, obj_id, obj_size = obj[0], obj[1], obj[2]

        # create the current state for cache simulator
        cost = 0

        # simulation
        # if the object size is larger than cache size
        if obj_size >= self.cache_size:
            # record the request
            cost += obj_size
            hit = 0
            try:
                self.non_cache[obj_id][1] = req
            except IndexError:
                self.non_cache[obj_id] = [obj_size, req]

        else:
            #  Search the object in the cache
            #  If hit
            try:
                self.cache[obj_id][1] = req
                self.count_bhr += obj_size
                self.count_ohr += 1
                hit = 1
                cost += obj_size

            #  If not hit
            except IndexError:
                # accept request
                if action == 1:
                    # find the object in the cache, no cost, OHR and BHR ++
                    # can't find the object in the cache, add the object into cache after replacement, cost ++
                    while cache_size_online_remain < obj_size:
                        rm_id = self.cache_pq[0][1]
                        cache_size_online_remain += self.cache_pq[0][0]
                        cost += self.cache_pq[0][0]
                        discard_obj_if_admit.append(rm_id)
                        heapq.heappop(self.cache_pq)
                        del self.cache[rm_id]

                    # add into cache
                    self.cache[obj_id] = [obj_size, req]
                    heapq.heappush(self.cache_pq, (obj_size, obj_id))
                    cache_size_online_remain -= obj_size

                    # cost value is based on size, can be changed
                    cost += obj_size
                    hit = 0

                # reject request
                else:
                    hit = 0
                    # record the request to non_cache
                    try:
                        self.non_cache[obj_id][1] = req
                    except IndexError:
                        self.non_cache[obj_id] = [obj_size, req]

        self.size_all += obj_size 
        bhr = float(self.count_bhr / self.size_all)
        ohr = float(self.count_ohr / (req + 1))
        # print("debug:", bhr, ohr)
        reward = hit * cost

        self.req += 1
        self.cache_remain = cache_size_online_remain

        
        info = [self.count_bhr, self.size_all, float(float(self.count_bhr) / float(self.size_all))]
        return reward, info

    def next_hit(self, obj):
        try:
            obj_id = obj[1]
            self.cache[obj_id][1] = self.cache[obj_id][1]
            return True

        except IndexError:
            return False

    def get_state(self, obj=[0, 0, 0, 0]):
        '''
        Return the state of the object,  [obj_size, cache_size_online_remain, recency (steps since object was last visited) = req - last visited time]
        If an object has never been seen before, assigned a constant for the recency feature.
        For more information, see Learning Caching Policies with Subsampling:
            http://mlforsystems.org/assets/papers/neurips2019/learning_wang_2019.pdf
        '''
        obj_time, obj_id, obj_size = obj[0], obj[1], obj[2]
        try:
            req = self.req - self.cache[obj_id][1]
        except IndexError:
            try:
                req = self.req - self.non_cache[obj_id][1]
            except IndexError:
                # Unseen objects (not in non_cache or cache) are assigned this recency constant
                req = config.cache_unseen_recency
        state = [obj_size, self.cache_remain, req]

        return state

class CacheEnv(core.Env):
    """
    Cache description.

    * STATE *
        The state is represented as a vector:
        [request object size, 
         cache remaining size, 
         time of last request to the same object]

    * ACTIONS *
    TODO: should be fixed here, there should be both 
        Whether the cache accept the incoming request, represented as an
        integer in [0, 1].

    * REWARD * (BHR)
        Cost of previous step (object size) * hit

    * REFERENCE *
    """
    def __init__(self, seed=42):
        self.seed(seed)
        self.cache_size = config.cache_size

        # load trace, attach initial online feature values
        self.src = TraceSrc(trace=config.cache_trace, cache_size=self.cache_size)

        # set up the state and action space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(self.src.min_values, \
                                      self.src.max_values, \
                                      dtype=np.float32)

        # cache simulator
        self.sim = CacheSim(cache_size=self.cache_size, \
                            policy='lru', \
                            action_space=self.action_space, \
                            state_space=self.observation_space)

        # reset environment (generate new jobs)
        self.reset(1, 2)

    def reset(self, low=0, high=1000):
        new_trace = self.np_random.randint(low, high)
        self.src.reset(new_trace)
        self.sim.reset()
        if config.cache_trace == 'test':
            print("New Env Start", new_trace)
        elif config.cache_trace == 'real':
            print("New Env Start Real")
        return self.sim.get_state()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def step(self, action):
        # 0 <= action < num_servers
        global accept
        assert self.action_space.contains(action)
        state, done = self.src.step()
        reward, info = self.sim.step(action, state)
        obj, done = self.src.next()
        while self.sim.next_hit(obj):
            state, done = self.src.step()
            hit_reward, info = self.sim.step(accept, state)
            reward += hit_reward
            if done is True:
                break
            obj, done = self.src.next()

        obs = self.sim.get_state(obj)
        info = {}
        return obs, reward, done, info


    def render(self, mode='human', close=False):
        pass
