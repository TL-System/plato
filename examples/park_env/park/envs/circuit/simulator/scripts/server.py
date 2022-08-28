import argparse
import asyncio
import inspect

from park.envs.circuit.simulator.circuit import exported_circuits, AsyncLocalContext
from park.envs.circuit.simulator.utility.comm import RobustServer
from park.envs.circuit.simulator.utility.io import loads_pickle, dumps_pickle
from park.envs.circuit.simulator.utility.logging import get_default_logger


class Manager(object):
    def __init__(self, path, logger=None):
        self._logger = logger or get_default_logger('CircuitServer')
        self._context = AsyncLocalContext(path)
        self._cache = dict()

    def _decode(self, name, method, values, debug):
        name = name.decode('utf-8')
        if name not in exported_circuits:
            raise NotImplementedError(f'Circuit "{name}" is not implemented or exported.')

        circuit = self._cache.get(name, exported_circuits[name])
        if inspect.isclass(circuit):
            circuit = circuit()
        self._cache.setdefault(name, circuit)

        method = method.decode('utf-8')
        values = loads_pickle(values)
        debug = loads_pickle(debug)
        return circuit, method, values, debug

    async def handler(self, name, method, values, debug):
        circuit, method, values, debug = self._decode(name, method, values, debug)
        self._logger.info(f'Simulating circuit {circuit}...')
        if method == 'simulate':
            result = await asyncio.wrap_future(circuit.evaluate(values, debug))
        elif method == 'simulate_batch':
            result = await asyncio.wrap_future(circuit.evaluate_batch(values, debug))
        else:
            raise NotImplementedError(f'Circuit method "{method}" is not implemented.')
        self._logger.info(f'Simulation complete {circuit}.')
        return dumps_pickle(result)

    def __enter__(self):
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./tmp')
    parser.add_argument('--port', type=int, default=10000)
    config = parser.parse_args()

    logger = get_default_logger('CircuitServer')
    header = f'Server is listening on port {config.port} at {config.path}\n'
    logger.info(header + ''.join(['==> ' + key + '\n' for key in exported_circuits]))

    with Manager(config.path) as manager, RobustServer() as server:
        server.bind('tcp', port=config.port)
        server.mainloop(manager.handler)
