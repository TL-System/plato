import datetime

__all__ = ['format_table', 'format_time']


def format_table(keys, values):
    max_key = max(map(len, keys))
    return [k + ': ' + ' ' * (max_key - len(k)) + str(v) for k, v in zip(keys, values)]


def format_time(interval):
    return str(datetime.timedelta(seconds=round(interval)))
