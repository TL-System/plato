#!/usr/bin/env python
"""
Test running a lot of selected configs.
This is for testing staleness bound.
"""

import subprocess
import time
import os

if __name__ == '__main__':
    for t in [
            'configs/port/stal_0',
            'configs/port/stal_2',
            'configs/port/stal_4',
            'configs/port/stal_6',
    ]:
        print(f"Running config: {t}.yml")
        log_root = '/content/drive/MyDrive/plato_staleness'
        log_folder = os.path.join(log_root,
                                  t.replace("/", "__").replace(".", ""))
        log_name = time.strftime("%Y_%m_%d__%H_%M_%S.txt", time.localtime())
        try:
            os.mkdir(log_root)
        except FileExistsError:
            pass
        try:
            os.mkdir(log_folder)
        except FileExistsError:
            pass
        with open(os.path.join(log_folder, log_name), 'a') as fp:
            fp.write(f'{t}\n\n')
            fp.flush()
            subprocess.call(['./run', f'--config={t}.yml'],
                            stdout=fp,
                            stderr=fp)
