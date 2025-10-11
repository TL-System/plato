"""
Utility functions that write results into a CSV file.
"""

import csv
import os
from typing import List


def initialize_csv(result_csv_file: str, logged_items: List, result_path: str) -> None:
    """Create a CSV file and writer the first row."""
    # Create a new directory if it does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_csv_file, "w", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        header_row = logged_items
        result_writer.writerow(header_row)


def write_csv(result_csv_file: str, new_row: List) -> None:
    """Write the results of current round."""
    with open(result_csv_file, "a", encoding="utf-8") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(new_row)
