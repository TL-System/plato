"""
Convert the result file to desired format
 and then plot the figure

"""

import os

import numpy as np

from libs.utils import data_to_format
from libs.plot_libs.box_plot import plot_group_boxes

file_path = "results/52002_accuracy.csv"

formatted_per_acc_df = data_to_format.personalized_acc_to_format(
    file_path=file_path)

rounds_idx = formatted_per_acc_df["round"].tolist()

rounds_samples = [
    formatted_per_acc_df.iloc[round_idx, 2:].dropna().to_numpy()
    for round_idx in range(len(rounds_idx))
]
rounds_position = [round_num for round_num in rounds_idx]
rounds_labels = rounds_position

plot_group_boxes(groups_samples=[rounds_samples],
                 groups_colors=[["red"]],
                 groups_boxes_positions=[rounds_position],
                 center_pos=rounds_position,
                 groups_label=["best"],
                 xticklabels=rounds_labels,
                 fig_style="seaborn-paper",
                 font_size="xx-small",
                 save_file_path=None,
                 save_file_name=None)
