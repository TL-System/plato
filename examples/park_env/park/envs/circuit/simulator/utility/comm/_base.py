import abc
import contextlib

import zmq

__all__ = ['_AbstractNode', 'socket_bind', 'socket_unbind', 'socket_connect', 'socket_disconnect']


class _AbstractNode(object, metaclass=abc.ABCMeta):
    def __init__(self):
        self._context = None
        self.__internal_context = None

    def initialize(self, context: zmq.Context = None):
        if self.__internal_context:
            raise ValueError("Node has already been activated.")
        if context:
            self.__internal_context = False
            return context
        self.__internal_context = True
        self._context = zmq.Context().instance()

    def finalize(self):
        if self.__internal_context:
            self._context.term()
        self._context = None
        self.__internal_context = None

    @property
    def context(self) -> zmq.Context:
        return self._context

    @contextlib.contextmanager
    def on_context(self, context) -> '_AbstractNode':
        try:
            self.initialize(context)
            yield self
        finally:
            self.finalize()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def socket_bind(socket: zmq.Socket, protocol, interface=None, port=None):
    if protocol == 'tcp':
        assert interface is None
        if port is None:
            port = socket.bind_to_random_port('tcp://0.0.0.0')
        else:
            socket.bind(f'tcp://0.0.0.0:{port}')
        return port
    else:
        assert interface is not None
        socket.bind(f'tcp://{interface}')


def socket_unbind(socket: zmq.Socket, protocol, interface=None, port=None):
    if protocol == 'tcp':
        assert port is not None and interface is None
        socket.unbind(f'tcp://0.0.0.0:{port}')
    else:
        assert interface is not None
        socket.unbind(f'tcp://{interface}')


def socket_connect(socket: zmq.Socket, protocol, interface, port=None):
    if protocol == 'tcp':
        assert port is not None
        socket.connect(f'{protocol}://{interface}:{port}')
    else:
        assert port is None
        socket.connect(f'{protocol}://{interface}')


def socket_disconnect(socket: zmq.Socket, protocol, interface, port=None):
    if protocol == 'tcp':
        assert port is not None
        socket.disconnect(f'{protocol}://{interface}:{port}')
    else:
        assert port is None
        socket.disconnect(f'{protocol}://{interface}')
