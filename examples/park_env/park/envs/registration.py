# Format follows OpenAI gym https://gym.openai.com

import re
import pkg_resources
from park import logger


def load(entry_point):
    import pkg_resources # takes ~400ms to load, so we import it lazily
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(entry_point))
    result = entry_point.resolve()
    return result


class EnvSpec(object):
    """
    A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.
    Args:
        id (str): The environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
    """

    def __init__(self, env_id, entry_point=None):
        self.env_id = env_id
        self._entry_point = entry_point

    def make(self):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self._entry_point is None:
            raise error.Error('Environment ' + self.env_id + ' needs to specify an entry point')
        elif callable(self._entry_point):
            env = self._entry_point()
        else:
            cls = load(self._entry_point)
            env = cls()

        return env


class EnvRegistry(object):
    """
    Register an env by ID. The goal is that results on a particular
    environment should be comparable.
    """

    def __init__(self):
        self.env_specs = {}

    def make(self, env_id):
        logger.info('Making new env ' + env_id)
        spec = self.spec(env_id)
        env = spec.make()
        return env

    def all(self):
        return self.env_specs.values()

    def spec(self, env_id):
        if env_id not in self.env_specs:
            raise KeyError('Environment ' + env_id + ' not defined.')
        return self.env_specs[env_id]

    def register(self, env_id, entry_point=None):
        if env_id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(env_id))
        self.env_specs[env_id] = EnvSpec(env_id, entry_point)


# Global registry
registry = EnvRegistry()

def register(env_id, entry_point):
    return registry.register(env_id, entry_point)

def make(env_id):
    return registry.make(env_id)

def spec(env_id):
    return registry.spec(env_id)
