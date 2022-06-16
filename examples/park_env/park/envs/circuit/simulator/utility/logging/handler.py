import logging
import sys

from park.envs.circuit.simulator.utility.logging import StructuredFormatterBuilder

__all__ = ['get_console_handler', 'get_logfile_handler', 'get_callback_handler']


def get_console_handler(console='stdout', level=logging.DEBUG, formatter=None):
    formatter = formatter or StructuredFormatterBuilder().get_colorful_formatter()
    if isinstance(console, str):
        if console in ('stdout', 'out', 'o'):
            console = sys.stdout
        if console in ('stderr', 'err', 'e'):
            console = sys.stderr
    else:
        assert id(console) in (id(sys.stdout), id(sys.stderr)), 'Console must be stdout or stderr.'
    handler = logging.StreamHandler(stream=console)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    setattr(handler, '__console__', console)
    return handler


def get_logfile_handler(logfile, mode='a', encoding=None, delay=False, level=logging.DEBUG, formatter=None):
    formatter = formatter or StructuredFormatterBuilder().get_formatter()
    handler = logging.FileHandler(logfile, mode, encoding, delay)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    setattr(handler, '__logfile__', logfile)
    return handler


class _CallbackHandler(logging.Handler):
    def __init__(self, callback, level: int = logging.NOTSET):
        super().__init__(level)
        assert callable(callback)
        self._callback = callback

    def emit(self, record):
        try:
            msg = self.format(record)
            self._callback(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_callback_handler(callback, level=logging.DEBUG, formatter=None):
    formatter = formatter or StructuredFormatterBuilder().get_colorful_formatter()
    handler = _CallbackHandler(callback)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler
