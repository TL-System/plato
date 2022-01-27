import os

import csv

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_to_dict(result_csv_file, plot_pairs=["round", "accuracy"]):
    """Read a CSV file and write the values that need to be plotted
    into a dictionary."""
    result_dict = {}

    for pairs in plot_pairs:
        pair = [x.strip() for x in pairs.split('-')]
        for item in pair:
            if item not in result_dict:
                result_dict[item] = []

    with open(result_csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict:
                if item in (
                        'round',
                        'global_round',
                        'local_epochs',
                ):
                    result_dict[item].append(int(row[item]))
                else:
                    result_dict[item].append(float(row[item]))

    return result_dict


def align_results(results_X_values):
    min_x_value = min([X_value[-1] for X_value in results_X_values])
    stop_indexs = []
    for _, result_X in enumerate(results_X_values):
        stop_idx = 0
        for value_idx, value in enumerate(result_X):
            if value >= min_x_value:
                stop_idx = value_idx
                break
        stop_indexs.append(stop_idx)

    return stop_indexs


def plot_results():
    cur_dir = cur_dir = os.path.abspath(os.getcwd())

    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False)  #去掉右边框

    # plot round-accuracy

    plot_items = ["elapsed_time", "accuracy"]

    # plot result 1 - port
    result1_file_name = "result1.csv"
    result1_file_path = os.path.join(cur_dir, "results", result1_file_name)
    result1_dict = read_csv_to_dict(result1_file_path, plot_items)
    result1_X = result1_dict[plot_items[0]]
    result1_Y = result1_dict[plot_items[1]]

    # plot results 2 - fedavg
    result2_file_name = "result2.csv"
    result2_file_path = os.path.join(cur_dir, "results", result2_file_name)
    result2_dict = read_csv_to_dict(result2_file_path, plot_items)
    result2_X = result2_dict[plot_items[0]]
    result2_Y = result2_dict[plot_items[1]]

    # obtain the min length
    value_align_idxs = align_results(results_X_values=[result1_X, result2_X])

    plt.plot(result1_X[:value_align_idxs[0]],
             result1_Y[:value_align_idxs[0]],
             color="black",
             label="Port",
             linewidth=5)
    plt.plot(result2_X[:value_align_idxs[1]],
             result2_Y[:value_align_idxs[1]],
             color="black",
             label="FedAsync",
             linewidth=3)

    # plot results 3 - xxx

    title = "Comp"
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold')  #默认字体大小为12
    plt.xlabel("Elapsed time", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=13, fontweight='bold')

    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')

    jpg_name = ("./results/result1_result2_cmp.jpg")
    pdf_name = ("./results/result1_result2_cmp.pdf")
    plt.savefig(jpg_name, format='jpg')
    plt.savefig(pdf_name, format='pdf')


plot_results()