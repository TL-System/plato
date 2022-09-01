"""
The implementation of saving the plotted figs properly.


"""

import os

import numpy as np
import matplotlib.pyplot as plt


def save_fig(whole_fig, save_file_name=None, save_path=None):
    save_file_name = "current_plot" if save_file_name is None else save_file_name
    save_path = os.getcwd() if save_path is None else save_path
    whole_fig.tight_layout()
    to_save = os.path.join(save_path, save_file_name)
    print("Saving to ", to_save)
    plt.savefig(to_save + ".png", dpi=300)
    plt.savefig(to_save + ".jpg", dpi=300)
    plt.savefig(to_save + '.pdf', format='pdf')
