import contextlib
import copy
import logging

from park.envs.circuit.simulator.utility.logging.formatter import _FormatBuilder

__all__ = ['context_handler', 'context_formatter']


@contextlib.contextmanager
def context_formatter(logger: logging.Logger, format_builder: _FormatBuilder):
    handlers = copy.copy(logger.handlers)
    try:
        del logger.handlers[:]
        for handler in handlers:
            handler = copy.copy(handler)
            handler.setFormatter(format_builder.get_formatter(type(handler.formatter)))
            logger.handlers.append(handler)
        yield
    finally:
        del logger.handlers[:]
        logger.handlers.extend(handlers)


@contextlib.contextmanager
def context_handler(logger: logging.Logger, handler: logging.Handler):
    try:
        logger.addHandler(handler)
        yield logger
    finally:
        logger.removeHandler(handler)


@contextlib.contextmanager
def context_disable(logger: logging.Logger, handler_filter):
    assert callable(handler_filter)
    original = copy.copy(logger.handlers)
    try:
        logger.handlers[:] = filter(handler_filter, logger.handlers)
        yield logger
    finally:
        logger.handlers[:] = original


@contextlib.contextmanager
def context_level(logger: logging.Logger, level):
    original_level = logger.level
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(original_level)
