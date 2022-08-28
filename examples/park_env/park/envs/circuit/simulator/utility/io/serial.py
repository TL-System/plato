import json
import numpy as np
import dill

__all__ = ['load_dill', 'dump_dill',
           'loads_dill', 'dumps_dill',
           'load_json', 'dump_json',
           'loads_json', 'dumps_json',
           'load_pickle', 'dump_pickle',
           'loads_pickle', 'dumps_pickle',
           'load_txt', 'dump_txt']

try:
    import cPickle as pickle
except ImportError:
    import pickle


def dump_pickle(data, path, **kwargs):
    with open(path, 'wb') as writer:
        pickle.dump(data, writer, **kwargs)


def dumps_pickle(data, **kwargs):
    return pickle.dumps(data, **kwargs)


def dump_json(data, path, indent=4):
    with open(path, 'w') as writer:
        json.dump(data, writer, indent=indent, cls=JSONExtendedEncoder)


def dumps_json(data, indent=4):
    return json.dumps(data, indent=indent, cls=JSONExtendedEncoder)


def dump_dill(data, path):
    with open(path, 'wb') as writer:
        dill.dump(data, writer)


def dumps_dill(data):
    return dill.dumps(data)


def load_txt(path):
    with open(path, 'r') as reader:
        return reader.read()


def dump_txt(data, path):
    with open(path, 'w') as writer:
        writer.write(data)


class JSONExtendedEncoder(json.JSONEncoder):
    def default(self, o):
        # Reference: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        if hasattr(o, '__jsonify__'):
            return o.__jsonify__()
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super(JSONExtendedEncoder, self).default(o)


def load_pickle(path):
    with open(path, 'rb') as reader:
        return pickle.load(reader)


def loads_pickle(data):
    return pickle.loads(data)


def load_json(path):
    with open(path, 'r') as reader:
        json.load(reader)


def loads_json(data):
    return json.dumps(data)


def load_dill(path):
    with open(path, 'rb') as reader:
        return dill.load(reader)


def loads_dill(data):
    return dill.loads(data)
