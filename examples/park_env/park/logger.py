import logging
from park.param import config


if config.logging_level == 'debug':
    level = logging.DEBUG
elif config.logging_level == 'info':
    level = logging.INFO
elif config.logging_level == 'warning':
    level = logging.WARNING
elif config.logging_level == 'error':
    level = logging.ERROR
else:
    raise ValueError('Unknown logging level ' + config.logging_level)


if config.log_to == 'print':
    logging.basicConfig(level=level)
else:
    logging.basicConfig(filename=config.log_to, level=level)


def debug(msg):
    logging.debug(msg)


def info(msg):
    logging.info(msg)


def warn(msg):
    logging.warning(msg)


def error(msg):
    logging.error(msg)


def exception(msg, *args, **kwargs):
    logging.exception(msg, *args, **kwargs)
