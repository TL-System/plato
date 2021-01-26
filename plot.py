"""
A simple utility that plot figures of results as PDF files, stored in results/.
"""

import csv
import matplotlib.pyplot as plt

from config import Config


def read_csv_to_dict(result_csv_file):
    """Read CSV file and write numbers that need to be plot into a dictionary."""
    result_dict = {}

    plot_pairs = Config().results.plot
    plot_pairs = [x.strip() for x in plot_pairs.split(',')]

    for pairs in plot_pairs:
        pair = [x.strip() for x in pairs.split('&')]
        for item in pair:
            if item not in result_dict:
                result_dict[item] = []

    with open(result_csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for item in result_dict:
                if item == 'gloabl_round':
                    result_dict[item].append(int(row[item]))
                else:
                    result_dict[item].append(float(row[item]))

    return result_dict


def plot(x_label, x_value, y_label, y_value, figure_file_name):
    """Plot a figure."""
    fig, ax = plt.subplots()
    ax.plot(x_value, y_value)
    ax.set(xlabel=x_label, ylabel=y_label)
    fig.savefig(figure_file_name)


def plot_figures_from_dict(result_csv_file, result_dir):
    """Plot figures with dictionary of results."""
    result_dict = read_csv_to_dict(result_csv_file)

    plot_pairs = Config().results.plot
    plot_pairs = [x.strip() for x in plot_pairs.split(',')]

    for pairs in plot_pairs:
        figure_file_name = result_dir + pairs + '.pdf'
        pair = [x.strip() for x in pairs.split('&')]
        x_y_labels = []
        x_y_values = {}
        for item in pair:
            label = {
                'global_round': 'Global training round',
                'accuracy': 'Accuracy (%)',
                'training_time': 'Training time (s)',
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

    if Config().results:
        dataset = Config().trainer.dataset
        model = Config().trainer.model
        server_type = Config().server.type
        result_dir = f'./results/{dataset}/{model}/{server_type}/'
        result_csv_file = result_dir + 'result.csv'
        print(f"Plotting results located at {result_csv_file}.")
        plot_figures_from_dict(result_csv_file, result_dir)
    else:
        print("No results to be plotted according to the configuration file.")


if __name__ == "__main__":
    main()
