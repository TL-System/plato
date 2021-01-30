"""
Utility functions that write results into a CSV file.
"""

import csv
import os
from config import Config


def initialize_csv(result_csv_file, recorded_items, result_dir):
    """Create a CSV file and writer the first row."""
    # Create a new directory if it does not exist.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_csv_file, 'w', newline='') as result_file:
        result_writer = csv.writer(result_file)
        first_row = recorded_items
        result_writer.writerow(first_row)

    if result_csv_file[-10:] == 'result.csv':
        # Use this if condition to avoid
        # repeatly writing note in cross-silo FL
        write_note(result_dir)


def write_note(result_dir):
    """Write note of this experiment"""
    note_file = result_dir + 'note.txt'
    note = open(note_file, 'w')
    note.write("This experiment uses configuration file: " +
               Config.args.config + '\n')
    note.write("Dataset: " + Config().data.dataset + '\n')
    note.write("ML model: " + Config().trainer.model + '\n')
    note.write("FL algorithm: " + Config().algorithm.type + '\n')
    note.write("Number of clients: " + str(Config.clients.total_clients) +
               '\n')
    if Config().algorithm.type == 'fedavg_cross_silo':
        note.write("Number of silos: " +
                   str(Config().algorithm.cross_silo.total_silos) + '\n')
        note.write("Number of edge aggregation rounds: " +
                   str(Config().algorithm.cross_silo.rounds) + '\n')
    note.close()


def write_csv(result_csv_file, new_row):
    """Write the results of current round."""
    with open(result_csv_file, 'a') as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(new_row)
