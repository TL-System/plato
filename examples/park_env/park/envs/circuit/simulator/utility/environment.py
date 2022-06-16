from park.core import Env
from park.envs.circuit.simulator.circuit import Evaluator
from park.envs.circuit.simulator.utility.misc import flatten


def l2dc_to_park_space(space):
    from park.envs.circuit.simulator import Box as L2DCBox,
    from park.spaces import Box as ParkBox
    if isinstance(space, L2DCBox):
        return ParkBox(low=space.min_bound, high=space.max_bound, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, )


class CircuitEnv(Env):
    def __init__(self, evaluator: Evaluator, benchmark):
        self.observation_space = [s for s in flatten(evaluator.out_space)]
        self.action_space = None
        self._evaluator = evaluator
        self._benchmark = benchmark

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def benchmark(self):
        return self._benchmark

    def step(self, action):
        pass

    def reset(self):
        self._running_values =
