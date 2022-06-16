import abc
import contextlib
import os
import subprocess
import traceback

import numpy as np

from park.envs.circuit.simulator.circuit import Context, LocalContext
from park.envs.circuit.simulator.utility.logging import get_default_logger
from park.spaces import Box

__all__ = ['Circuit', 'Evaluator', 'export_circuit', 'exported_circuits', 'make_circuit']

exported_circuits = dict()


def export_circuit(cls):
    assert cls.__name__ not in exported_circuits
    exported_circuits.setdefault(cls.__name__, cls)
    return cls


def make_circuit(name, *args, **kwargs):
    if name not in exported_circuits:
        raise ValueError(f'"{name}" is not registered as circuit')
    return exported_circuits[name](*args, **kwargs)


class Circuit(object, metaclass=abc.ABCMeta):
    __UNITS__ = {'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'x': 1e6, 'g': 1e9}

    def __init__(self, default_context=None):
        self._default_context = default_context or LocalContext()

    @property
    @abc.abstractmethod
    def parameters(self) -> tuple:
        pass

    @abc.abstractmethod
    def run(self, tmp_path, values):
        pass

    def evaluate(self, values, debug=None):
        context = contextlib.suppress() if Context.any_opened() else self._default_context
        with context:
            return Context.current_context().evaluate(self, values, debug)

    def evaluate_batch(self, values, debug=None):
        context = contextlib.suppress() if Context.any_opened() else self._default_context
        with context:
            return Context.current_context().evaluate_batch(self, values, debug)

    @staticmethod
    def _run_hspice(name, work_path) -> str:
        pipe = subprocess.Popen(['hspice', name + ' > result'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_path)
        pipe.communicate()
        with open(os.path.join(work_path, 'result'), 'r') as reader:
            return reader.read()

    @staticmethod
    def _run_spectre(name, work_path) -> str:
        pipe = subprocess.Popen(['spectre', name, '-format=psfascii'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work_path)
        pipe.communicate()
        with open(os.path.join(work_path, 'spectre.dc')) as reader:
            return reader.read()

    @classmethod
    def number_from_string(cls, string: str):
        string = string.strip()
        for unit, value in cls.__UNITS__.items():
            if string.endswith(unit):
                return eval(string[:-1]) * value
        return eval(string)

    def evaluator(self, **kwargs):
        return Evaluator(self, **kwargs)

    @property
    @abc.abstractmethod
    def out_space(self):
        pass

    # @property
    # def in_space(self):
    #     return Box(shape=len(self.parameters))


class Evaluator(object):
    def __init__(self, circuit: Circuit, logger=None):
        super(Evaluator, self).__setattr__('_circuit', circuit)
        super(Evaluator, self).__setattr__('_preset', {})
        super(Evaluator, self).__setattr__('_lbound', {})
        super(Evaluator, self).__setattr__('_ubound', {})
        super(Evaluator, self).__setattr__('_logger', logger or get_default_logger(self.__class__.__name__))

    @property
    def circuit(self):
        return self._circuit

    @property
    def lower_bound(self):
        return [self._lbound[k] for k in self.parameters]

    @property
    def upper_bound(self):
        return [self._ubound[k] for k in self.parameters]

    @property
    def bound(self):
        return self.lower_bound, self.upper_bound

    def set_bound(self, parameter, lower_bound, upper_bound):
        if lower_bound is not None:
            self._lbound[parameter] = lower_bound
        if upper_bound is not None:
            self._ubound[parameter] = upper_bound

    def denormalize(self, param, value, source_bound=(0, 1)):
        value = (value - source_bound[0]) / (source_bound[1] - source_bound[0])
        return value * (self._ubound[param] - self._lbound[param]) + self._lbound[param]

    def normalize(self, param, value, target_bound=(0, 1)):
        value = (value - self.lower_bound[param]) / (self._ubound[param] - self._lbound[param])
        return value * (target_bound[1] - target_bound[0]) + target_bound[0]

    def random_values(self, np_state=None):
        np_state = np_state or np.random
        return tuple(self.denormalize(k, np_state.rand()) for k in self.parameters)

    def sample(self, debug=None):
        values = self.random_values()
        return self(values, debug)

    def sample_batch(self, size, debug=None):
        values = [self.random_values() for _ in range(size)]
        return self.batch(values, debug)

    @property
    def parameters(self) -> tuple:
        return tuple([i for i in self._circuit.parameters if i not in self._preset])

    @property
    def out_space(self):
        return self._circuit.out_space

    @property
    def in_space(self):
        space = Box(low=np.asarray(self.lower_bound), high=np.asarray(self.upper_bound), dtype=np.float32)
        # assert space.shape == self._circuit.in_space.shape
        return space

    def __setitem__(self, key, value):
        if key in self.parameters:
            self._preset[key] = value
        else:
            raise ValueError(f'No such parameter "{key}"')

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def formalize(self, values):
        if isinstance(values, dict):
            values.update(self._preset)
            assert set(values.keys()) == set(self.parameters)
            return values
        else:
            values = tuple(values)
            params = [k for k in self.parameters if k not in self._preset]
            assert len(values) == len(params)
            values = dict(zip(params, values))
            values.update(self._preset)
            return values

    def formalize_as_numpy_array(self, values):
        values = self.formalize(values)
        return np.asarray([values[key] for key in self.parameters])

    def __call__(self, values, debug=None):
        values = self.formalize(values)
        try:
            return self._circuit.evaluate(values, debug)
        except KeyboardInterrupt:
            raise
        except:
            self._logger.exception("An exception occurred when evaluate single value.")
            return None

    def batch(self, values, debug=None):
        values = tuple(map(self.formalize, values))
        values = self._circuit.evaluate_batch(values, debug)
        for i in range(len(values)):
            if isinstance(values[i], Exception):
                self._logger.exception(f"An exception occurred when evaluate batch value {i}.",
                                       exc_info=traceback.extract_tb(values[i].__traceback__))
                values[i] = None
        return values
