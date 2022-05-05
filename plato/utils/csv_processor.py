"""
Utility functions that write results into a CSV file.
"""

import csv
import os
from typing import List


def initialize_csv(result_csv_file: str, recorded_items: List,
                   result_dir: str) -> None:
    """Create a CSV file and writer the first row."""
    # Create a new directory if it does not exist.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_csv_file, 'w', encoding='utf-8',
              newline='') as result_file:
        result_writer = csv.writer(result_file)
        first_row = recorded_items
        result_writer.writerow(first_row)

def initialize_accuracy_csv(result_csv_file: str, num_rounds: List, num_clients: List,
                   result_dir: str) -> None:
    """Creates csv file for the test accuracy"""
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_csv_file, 'w', encoding='utf-8',
              newline='') as result_file:

        result_writer = csv.writer(result_file)
        first_row = num_rounds
        result_writer.writerow(first_row)

        for i in range(len(num_clients) - 1):
            result_writer.writerow([num_clients[i + 1]])


def write_csv(result_csv_file: str, new_row: List) -> None:
    """ Write the results of current round. """
    with open(result_csv_file, 'a', encoding='utf-8') as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(new_row)

def write_csv_column(result_csv_file: str, new_column: List) -> None:
    """ Writes the test accuracies of current round. """
    with open(result_csv_file, 'r', encoding='utf-8') as result_file:
        csv_reader = csv.reader(result_file)
        data = list(csv_reader)
        data = [x for x in data if x != []]
        for i in range(1, len(data)):
            data[i].append(new_column[i - 1])


    with open(result_csv_file, 'w', encoding='utf-8') as result_file:
        result_writer = csv.writer(result_file)
        for i in range(len(data)):
            result_writer.writerow(data[i])