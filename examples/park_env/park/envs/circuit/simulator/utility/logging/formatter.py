import abc
import copy
import logging

import termcolor

__all__ = ['_FormatBuilder', 'StructuredFormatterBuilder', 'FormatBuilder']


class ColorfulFormatter(logging.Formatter):
    @staticmethod
    def colored_level(level):
        if level == 'CRITICAL':
            return termcolor.colored(level, 'white', 'on_red')
        elif level == 'ERROR':
            return termcolor.colored(level, 'red')
        elif level == 'WARNING' or level == 'WARN':
            return termcolor.colored(level, 'yellow')
        elif level == 'INFO':
            return termcolor.colored(level, 'blue')
        elif level == 'DEBUG':
            return termcolor.colored(level, 'grey')
        else:
            return level

    @staticmethod
    def colored_time(time):
        return termcolor.colored(time, 'green')

    @staticmethod
    def colored_name(name):
        return termcolor.colored(name, 'magenta')

    @staticmethod
    def colored_process(process):
        return termcolor.colored(process, 'blue')

    @staticmethod
    def colored_thread(thread):
        return termcolor.colored(thread, 'yellow')

    @staticmethod
    def colored_lineno(lineno):
        return termcolor.colored(lineno, 'yellow')

    @staticmethod
    def colored_funcname(funcname):
        return termcolor.colored(funcname, 'cyan')

    @staticmethod
    def colored_pathname(pathname):
        return termcolor.colored(pathname, 'white', attrs=['underline'])

    def formatTime(self, record, datefmt=None):
        time = super(ColorfulFormatter, self).formatTime(record, datefmt)
        return self.colored_time(time)

    def format(self, record):
        record = copy.copy(record)
        record.levelname = self.colored_level(record.levelname)
        record.levelno = self.colored_level(record.levelno)
        record.name = self.colored_name(record.name)
        record.process = self.colored_process(record.process)
        record.processName = self.colored_process(record.processName)
        record.thread = self.colored_thread(record.thread)
        record.threadName = self.colored_thread(record.threadName)
        record.lineno = self.colored_lineno(record.lineno)
        record.pathname = self.colored_pathname(record.pathname)
        record.funcName = self.colored_funcname(record.funcName)
        return super(ColorfulFormatter, self).format(record)


class _FormatBuilder(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_fmt(self):
        pass

    @abc.abstractmethod
    def get_datefmt(self):
        pass

    def get_formatter(self, formatter_type=logging.Formatter):
        return formatter_type(self.get_fmt(), self.get_datefmt())

    def get_colorful_formatter(self):
        return self.get_formatter(ColorfulFormatter)


class FormatBuilder(_FormatBuilder):
    def __init__(self, fmt=None, datefmt=None):
        self._fmt = fmt
        self._datefmt = datefmt

    def get_fmt(self):
        return self._fmt

    def get_datefmt(self):
        return self._datefmt


class StructuredFormatterBuilder(_FormatBuilder):
    def __init__(self, use_time=True, use_name=True, use_funcname=False,
                 use_process=False, use_thread=False, use_level=True, datefmt=None):
        self.use_name = use_name
        self.use_funcname = use_funcname
        self.use_process = use_process
        self.use_thread = use_thread

        self.use_level = use_level
        self.datefmt = datefmt
        self.use_time = use_time

    def get_fmt(self):
        context = []
        self.use_time and context.append('%(asctime)s')
        self.use_name and context.append('%(name)s')
        self.use_funcname and context.append('%(funcName)s')
        self.use_process and context.append('%(processName)s')
        self.use_thread and context.append('%(threadName)s')

        footer = '%(levelname)s' if self.use_level else ''

        caption = []
        if len(context) > 0:
            if len(context) == 1:
                caption.append(context[0])
            else:
                caption.append(' '.join((context[0], '(' + ' '.join(context[1:]) + ')')))
        footer and caption.append(footer)
        caption = ' '.join(caption)
        if caption:
            return f'{caption}: %(message)s'
        else:
            return f'%(message)s'

    def get_datefmt(self):
        return self.datefmt if self.datefmt else '%Y-%m-%d %H:%M:%S'
