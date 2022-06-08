import os

import numpy as np
import pandas as pd


def read_xlsx_data(data_dir, file_name):
    loaded_data = pd.read_excel(os.path.join(data_dir, file_name))
    return loaded_data


def read_npy_ldata(load_dir, load_name):
    save_path = os.path.join(load_dir, load_name)
    dt = np.load(save_path)
    return dt


def read_npy_data(load_dir, load_name):
    save_path = os.path.join(load_dir, load_name)
    dt = np.load(save_path)

    return dt[:, 0], dt[:, 1]


def read_npy_1ddata(load_dir, load_name):
    save_path = os.path.join(load_dir, load_name)
    dt = np.load(save_path)
    len_dt = int(len(dt))
    half_len = int(len_dt / 2)

    return dt[:half_len], dt[half_len:len_dt]
