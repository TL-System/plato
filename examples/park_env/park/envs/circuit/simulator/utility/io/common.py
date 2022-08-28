import contextlib
import datetime
import os
import shutil
import uuid

__all__ = ['get_extname', 'is_file_exist', 'file_remove', 'open_tmp_path', 'get_relpath']


def get_extname(path, match_first=False):
    if match_first:
        path = os.path.split(path)[1]
        return path[path.find('.'):]
    else:
        return os.path.splitext(path)[1]


def is_file_exist(path):
    return os.path.isfile(path)


def file_remove(path):
    if is_file_exist(path):
        os.remove(path)


def get_relpath(path, compare_path=None):
    compare_path = compare_path or os.getcwd()
    return os.path.relpath(path, compare_path)


@contextlib.contextmanager
def open_tmp_path(base_path, tmp_type='time', keep_folder='none'):
    assert keep_folder in ('none', 'onerror', 'always')
    if tmp_type == 'time':
        tmp_name = str(datetime.datetime.now())
    elif tmp_type == 'timepid':
        tmp_name = str(datetime.datetime.now()) + '-' + str(os.getpid())
    elif tmp_type == 'uuid':
        tmp_name = str(uuid.uuid4())
    else:
        raise ValueError(f'Cannot use temporary type of "{tmp_type}"')
    tmp_path = os.path.join(base_path, tmp_name or str(uuid.uuid4()))
    os.makedirs(tmp_path, exist_ok=True)
    try:
        yield tmp_path
    except Exception:
        if keep_folder != 'onerror':
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    if keep_folder != 'always':
        shutil.rmtree(tmp_path, ignore_errors=True)
