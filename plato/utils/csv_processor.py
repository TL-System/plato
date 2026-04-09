"""
Utility functions that write results into a CSV file.
"""

import csv
import os
from typing import List


def initialize_csv(result_csv_file: str, logged_items: list, result_path: str) -> None:
    """Create a CSV file and writer the first row."""
    # Create a new directory if it does not exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_csv_file, "w", encoding="utf-8", newline="") as result_file:
        result_writer = csv.writer(result_file)
        header_row = logged_items
        result_writer.writerow(header_row)


def expand_csv_columns(result_csv_file: str, additional_columns: list[str]) -> None:
    """Append new header columns and pad existing rows when the schema grows."""
    if not additional_columns or not os.path.exists(result_csv_file):
        return

    with open(result_csv_file, encoding="utf-8", newline="") as result_file:
        rows = list(csv.reader(result_file))

    if not rows:
        return

    header = rows[0]
    new_columns = [column for column in additional_columns if column not in header]
    if not new_columns:
        return

    rows[0] = header + new_columns
    for row_index in range(1, len(rows)):
        rows[row_index].extend([""] * len(new_columns))

    with open(result_csv_file, "w", encoding="utf-8", newline="") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerows(rows)


def write_csv(result_csv_file: str, new_row: list) -> None:
    """Write the results of current round."""
    with open(result_csv_file, "a", encoding="utf-8", newline="") as result_file:
        result_writer = csv.writer(result_file)
        result_writer.writerow(new_row)
