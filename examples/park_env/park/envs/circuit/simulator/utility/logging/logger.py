import logging

from park.envs.circuit.simulator.utility.logging import StructuredFormatterBuilder, get_console_handler

__all__ = ['get_logger', 'get_default_logger']


def get_logger(name, *handlers: logging.Handler, level=logging.DEBUG, propagate=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    del logger.handlers[:]
    logger.handlers.extend(handlers)
    logger.propagate = propagate
    return logger


def get_default_logger(name, *, level=logging.DEBUG, propagate=True, **kwargs):
    basic_handler = get_console_handler(formatter=StructuredFormatterBuilder(**kwargs).get_colorful_formatter())
    return get_logger(name, basic_handler, level=level, propagate=propagate)
