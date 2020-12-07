"""
Functions that plot figures of results.
"""

import os
import matplotlib.pyplot as plt


def plotting(x_value, x_label, y_value, y_label, figure_path, figure_name):
    """Plot a figure"""
    fig, ax = plt.subplots()
    ax.plot(x_value, y_value)
    ax.set(xlabel=x_label, ylabel=y_label)
    create_folder(figure_path)
    fig.savefig(figure_path + '/' + figure_name + '.pdf')


def plot_global_round_vs_accuracy(accuracy_list, figure_path):
    """Plot figure of global training round vs glboal model accuracy."""
    x_value = [i+1 for i in range(len(accuracy_list))]
    y_value = accuracy_list
    x_label = 'Global training round'
    y_label = 'Accuracy (%)'
    figure_name = 'global_round_vs_accuracy'
    plotting(x_value, x_label, y_value, y_label, figure_path, figure_name)

def plot_local_epoch_vs_accuracy(accuracy_list, local_epoch_num_list, figure_path):
    """Plot figure of local training epoch vs glboal model accuracy."""
    x_value = [sum(local_epoch_num_list[:i+1]) for i in range(len(local_epoch_num_list))]
    y_value = accuracy_list
    x_label = 'Local training epoch'
    y_label = 'Accuracy (%)'
    figure_name = 'local_epoch_vs_accuracy'
    plotting(x_value, x_label, y_value, y_label, figure_path, figure_name)

def plot_training_time_vs_accuracy(accuracy_list, training_time_list, figure_path):
    """Plot figure of training time vs glboal model accuracy."""
    x_value = [sum(training_time_list[:i+1]) for i in range(len(training_time_list))]
    y_value = accuracy_list
    x_label = 'Training time (s)'
    y_label = 'Accuracy (%)'
    figure_name = 'training_time_vs_accuracy'
    plotting(x_value, x_label, y_value, y_label, figure_path, figure_name)

def create_folder(folder_name):
    """Create folder if it does not exist"""
    exist = os.path.exists(folder_name)   
    if not exist:
        # Create folder
        os.makedirs(folder_name)
