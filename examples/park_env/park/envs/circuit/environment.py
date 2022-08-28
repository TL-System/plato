import numpy as np

from park.core import Env
from park.envs.circuit.simulator.circuit import Evaluator
from park.envs.circuit.simulator.utility.misc import nested_select, ordered_flatten
from park.spaces import Tuple


class CircuitEnv(Env):
    def __init__(self, evaluator: Evaluator, benchmark, obs_mark):
        self._evaluator = evaluator
        self._benchmark = benchmark
        self._obs_mark = obs_mark
        self.observation_space = ordered_flatten(nested_select(self._evaluator.out_space, obs_mark)[0])
        self.observation_space = Tuple(self.observation_space)
        self.action_space = self._evaluator.in_space

    def _reset_internal_state(self):
        self._running_param = []
        self._running_score = []

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def zero_obs(self):
        if isinstance(self.observation_space, Tuple):
            return tuple(np.zeros(space.shape) for space in self.observation_space.spaces)
        else:
            return np.zeros(self.observation_space.shape)


class CircuitPointedEnv(CircuitEnv):
    def __init__(self, evaluator: Evaluator, benchmark, obs_mark, total_steps):
        super().__init__(evaluator, benchmark, obs_mark)
        self._total_steps = total_steps

    def reset(self):
        self._reset_internal_state()
        return self.zero_obs()

    def step(self, action):
        features = self._evaluator(action)
        score = self._benchmark(action, features)

        self._running_param.append(action)
        self._running_score.append(score)

        now_score = self._running_score[-1]
        old_score = self._running_score[-2] if len(self._running_score) > 1 else 0
        reward = now_score - old_score

        obs, info = nested_select(features, self._obs_mark)
        obs = ordered_flatten(obs)

        return obs, reward, len(self._running_score) == self._total_steps, info
