import abc
import concurrent.futures
import functools
import os

from park.envs.circuit.simulator.utility.comm import RobustClient
from park.envs.circuit.simulator.utility.concurrency import make_pool
from park.envs.circuit.simulator.utility.io import open_tmp_path, loads_pickle, dumps_pickle
from park.envs.circuit.simulator.utility.logging import get_default_logger
from park.envs.circuit.simulator.utility.misc import AttrDict

__all__ = ['Context', 'RemoteContext', 'AsyncLocalContext', 'LocalContext']


class Context(object, metaclass=abc.ABCMeta):
    __current_context = []

    def __init__(self, debug='onerror'):
        self._debug = debug
        self.__opened = False

    @staticmethod
    def _evaluate(path, circuit, values, debug, no_except=False):
        base_path = os.path.join(path, circuit.__class__.__name__)
        with open_tmp_path(base_path, 'timepid', debug) as path:
            try:
                result = circuit.run(path, AttrDict(**values))
                if debug:
                    result.tmp_path = path
                return result
            except Exception as e:
                wrapped_exception = RuntimeError(f"Circuit simulation error (path={path})")
                wrapped_exception.__cause__ = e
                if no_except:
                    return wrapped_exception
                raise wrapped_exception

    @abc.abstractmethod
    def evaluate(self, circuit, values, debug=None):
        pass

    @abc.abstractmethod
    def evaluate_batch(self, circuit, values, debug=None):
        pass

    @classmethod
    def current_context(cls) -> 'Context':
        if not cls.__current_context:
            raise ValueError('No context is opened.')
        return cls.__current_context[-1]

    @classmethod
    def any_opened(cls):
        return len(cls.__current_context) > 0

    @property
    def opened(self):
        return self.__opened

    def __enter__(self):
        assert not self.__opened, "Context cannot be reopened."
        self.__current_context.append(self)
        self.__opened = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__opened = False
        self.__current_context.pop()

    def __repr__(self):
        return self.__str__()


class LocalContext(Context):
    def __init__(self, path='./tmp', debug='onerror'):
        super().__init__(debug)
        self._path = path
        self._pool = None

    @property
    def path(self):
        return self._path

    def __enter__(self):
        super(LocalContext, self).__enter__()
        self._pool = make_pool('process', os.cpu_count())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.terminate()
        self._pool = None
        super(LocalContext, self).__exit__(exc_type, exc_val, exc_tb)

    def evaluate(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._evaluate(self._path, circuit, values, debug)

    def evaluate_batch(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        func = functools.partial(self._evaluate, self._path, circuit, debug=debug, no_except=True)
        return self._pool.map(func, values)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return f'{self.__class__.__name__}(path={self._path})'


class AsyncLocalContext(LocalContext):
    def evaluate(self, circuit, values, debug=None):
        future = concurrent.futures.Future()
        debug = self._debug if debug is None else debug
        self._pool.apply_async(self._evaluate, (self._path, circuit, values, debug),
                               callback=future.set_result, error_callback=future.set_exception)
        return future

    def evaluate_batch(self, circuit, values, debug=None):
        future = concurrent.futures.Future()
        debug = self._debug if debug is None else debug
        func = functools.partial(self._evaluate, self._path, circuit, debug=debug, no_except=True)
        self._pool.map_async(func, values, callback=future.set_result, error_callback=future.set_exception)
        return future


class RemoteContext(Context):
    def __init__(self, host, port, logger=None, debug='onerror'):
        super().__init__(debug)
        self._host = host
        self._port = port
        self._logger = logger or get_default_logger(self.__class__.__name__)
        self._client = RobustClient(self._logger)

    def __enter__(self):
        super(RemoteContext, self).__enter__()
        self._client.initialize()
        self._client.connect('tcp', self._host, self._port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.finalize()
        super(RemoteContext, self).__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _encode(circuit, method, values, debug):
        name = circuit.__class__.__name__.encode('utf-8')
        method = str(method).encode('utf-8')
        values = dumps_pickle(values)
        debug = dumps_pickle(debug)
        return name, method, values, debug

    def _request(self, name, method, values, debug):
        result, = self._client.request(name, method, values, debug)
        return loads_pickle(result)

    def evaluate(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._request(*self._encode(circuit, 'simulate', values, debug))

    def evaluate_batch(self, circuit, values, debug=None):
        debug = self._debug if debug is None else debug
        return self._request(*self._encode(circuit, 'simulate_batch', values, debug))

    def __getstate__(self):
        raise NotImplementedError("Cannot pickle RemoteContext object.")

    def __setstate__(self, state):
        raise NotImplementedError("Cannot pickle RemoteContext object.")

    def __str__(self):
        return f'{self.__class__.__name__}(host={self._host}, port={self._port})'
