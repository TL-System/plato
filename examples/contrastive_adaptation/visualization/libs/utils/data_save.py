import os

import numpy as np


def save_data(data, save_dir, save_name):
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, data)


def save_datasets(data_XY, save_dir, save_name):
    combined_dt = np.column_stack(data_XY)

    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, combined_dt)


def save_data_xlsx(data_df, data_dir, file_name):
    save_path = os.path.join(data_dir, file_name)
    data_df.to_csv(save_path, index=False)


def load_data(load_dir, file_name):
    load_path = os.path.join(load_dir, file_name)
    return np.load(load_path)