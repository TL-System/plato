"""
A simple utility that plot figures of results as PDF files, stored in results/.
"""

import csv
import os
from typing import Dict, List, Any
from matplotlib import markers

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plato.config import Config


def read_csv_to_dict(result_csv_file: str, x_item, y_item) -> Dict[str, List]:
    """Read a CSV file and write the values that need to be plotted
    into a dictionary."""
    result_dict: Dict[str, List] = {}
    result_dict[x_item] = []
    result_dict[y_item] = []
    #print("result_dict: ", result_dict)
    """
    plot_pairs = #Config().params['plot_pairs']
    plot_pairs = [x.strip() for x in plot_pairs.split(',')]

    for pairs in plot_pairs:
        pair = [x.strip() for x in pairs.split('-')]
        for item in pair:
            if item not in result_dict:
                result_dict[item] = []
    """

    with open(result_csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            #print("row: ", row)
            for item in result_dict:
                if item in (
                        'round',
                        'global_round',
                        'local_epochs',
                ):
                    #print("row[item]: ", row[item])
                    result_dict[item].append(int(row[item]))
                else:
                    result_dict[item].append(float(row[item]))

    return result_dict


def plot(x_label, x_value1, x_value2, x_value3, x_value4, y_label, y_value1,
         y_value2, y_value3, y_value4, figure_file_name):
    """Plot a figure."""
    fig, ax = plt.subplots()
    ax.plot(
        x_value1,
        y_value1,
        color='green',
        linewidth=1.5,
        linestyle='--',
        #marker='o',
        label='FedAvg')
    ax.plot(
        x_value2,
        y_value2,
        color='orange',
        linewidth=1.5,
        linestyle='-.',
        #marker='v',
        label='Oort')
    ax.plot(
        x_value3,
        y_value3,
        color='blue',
        linewidth=1.5,
        linestyle=':',
        #marker='s',
        label='FedBuff')
    ax.plot(
        x_value4,
        y_value4,
        color='red',
        linewidth=1.5,
        linestyle=':',
        #marker='s',
        label='Async_selection')
    ax.legend(loc='lower right')
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(linestyle=':')
    fig.savefig(figure_file_name)


def main():
    # load comparison files

    x_item = 'elapsed_time'
    y_item = 'accuracy'

    result_csv_file1 = './fedavg_cifar_concen5_rand2.csv'
    result_dict1 = read_csv_to_dict(result_csv_file1, x_item, y_item)

    result_csv_file2 = './oort_cifar_concen5_rand2.csv'
    result_dict2 = read_csv_to_dict(result_csv_file2, x_item, y_item)

    result_csv_file3 = './fedbuff_cifar_concen5_rand2.csv'
    result_dict3 = read_csv_to_dict(result_csv_file3, x_item, y_item)

    result_csv_file4 = './async_cifar_concen5_rand2.csv'
    result_dict4 = read_csv_to_dict(result_csv_file4, x_item, y_item)
    #x_item = 'round'
    x_label = 'Elapsed Time'
    x_value1 = result_dict1[x_item]
    x_value2 = result_dict2[x_item]
    x_value3 = result_dict3[x_item]
    x_value4 = result_dict4[x_item]

    #y_item = 'accuracy'
    y_label = 'Accuracy (%)'
    y_value1 = result_dict1[y_item]
    y_value2 = result_dict2[y_item]
    y_value3 = result_dict3[y_item]
    y_value4 = result_dict4[y_item]

    figure_file_name = './elapsedTime_accuracy_500_20_CIFAR10_rand2.pdf'

    plot(x_label, x_value1, x_value2, x_value3, x_value4, y_label, y_value1,
         y_value2, y_value3, y_value4, figure_file_name)


if __name__ == "__main__":
    main()
