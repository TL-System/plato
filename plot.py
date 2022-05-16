"""
A simple utility that plot figures of results as PDF files, stored in results/.
"""

import csv
import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from plato.config import Config


def read_csv_to_dict(result_csv_file: str) -> Dict[str, List]:
    """Read a CSV file and write the values that need to be plotted
    into a dictionary."""
    result_dict: Dict[str, List] = {}

    plot_pairs = Config().params['plot_pairs']
    plot_pairs = [x.strip() for x in plot_pairs.split(',')]

    for pairs in plot_pairs:
        pair = [x.strip() for x in pairs.split('-')]
        for item in pair:
            if item not in result_dict:
                result_dict[item] = []

    with open(result_csv_file, 'r', encoding='utf-8') as f:
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


def plot(x_label: str, x_value: List[Any], y_label: str, y_value: List[Any],
         figure_file_name: str):
    """Plot a figure."""
    fig, ax = plt.subplots()
    ax.plot(x_value, y_value)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(figure_file_name)


def plot_figures_from_dict(result_csv_file: str, result_path: str):
    """Plot figures with dictionary of results."""
    result_dict = read_csv_to_dict(result_csv_file)

    plot_pairs = Config().params['plot_pairs']
    plot_pairs = [x.strip() for x in plot_pairs.split(',')]

    for pairs in plot_pairs:
        figure_file_name = os.path.join(result_path, pairs + '.pdf')
        pair = [x.strip() for x in pairs.split('-')]
        x_y_labels: List = []
        x_y_values: Dict[str, List] = {}
        for item in pair:
            label = {
                'round': 'Round',
                'accuracy': 'Accuracy (%)',
                'elapsed_time': 'Wall clock time elapsed (s)',
                'round_time': 'Training time in each round (s)',
                'global_round': 'Global training round',
                'local_epoch_num': 'Local epochs',
                'edge_agg_num': 'Aggregation rounds on edge servers'
            }[item]
            x_y_labels.append(label)
            x_y_values[label] = result_dict[item]

        x_label = x_y_labels[0]
        y_label = x_y_labels[1]
        x_value = x_y_values[x_label]
        y_value = x_y_values[y_label]
        plot(x_label, x_value, y_label, y_value, figure_file_name)


def main():
    """Plotting figures from the run-time results."""
    __ = Config()

    result_csv_path = Config().params['result_path']
    result_csv_file = os.path.join(result_csv_path, [
        file for file in os.listdir(result_csv_path)
        if (file.endswith(".csv") and not file.endswith("accuracy.csv"))
    ][0])
    print(f"Plotting results located at {result_csv_file}.")
    plot_figures_from_dict(result_csv_file, Config().params['result_path'])


if __name__ == "__main__":
    main()
