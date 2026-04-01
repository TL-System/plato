"""
Defines the ServerCallback class, which is the abstract base class to be subclassed
when creating new server callbacks.

Defines a default callback to print training progress.
"""

import logging
import os
from abc import ABC

from plato.config import Config
from plato.utils import csv_processor, fonts


class ServerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new server callbacks.
    """

    def __init__(self):
        """
        Initializer.
        """

    def on_weights_received(self, server, weights_received):
        """
        Event called after the updated weights have been received.
        """

    def on_weights_aggregated(self, server, updates):
        """
        Event called after the updated weights have been aggregated.
        """

    def on_clients_selected(self, server, selected_clients, **kwargs):
        """
        Event called after a new client arrived.
        """

    def on_clients_processed(self, server, **kwargs):
        """Additional work to be performed after client reports have been processed."""

    def on_training_will_start(self, server, **kwargs):
        """
        Event called before selecting clients for the first round of training.
        """

    def on_server_will_close(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """


class LogProgressCallback(ServerCallback):
    """
    A callback which prints a message when needed.
    """

    def __init__(self):
        super().__init__()

        recorded_items = Config().params["result_types"]
        self.recorded_items = [x.strip() for x in recorded_items.split(",")]

        # Initialize the .csv file for logging runtime results
        result_csv_file = f"{Config().params['result_path']}/{os.getpid()}.csv"
        csv_processor.initialize_csv(
            result_csv_file, self.recorded_items, Config().params["result_path"]
        )

        logging.info(
            fonts.colourize(
                f"[{os.getpid()}] Logging runtime results to: {result_csv_file}."
            )
        )

    def _result_csv_file(self) -> str:
        """Return the runtime CSV file path for the current server process."""
        return f"{Config().params['result_path']}/{os.getpid()}.csv"

    def _ensure_evaluation_columns(self, logged_items: dict[str, object]) -> None:
        """Expand the CSV schema when new structured evaluation metrics appear."""
        additional_columns = [
            item
            for item in logged_items
            if item.startswith("evaluation_") and item not in self.recorded_items
        ]
        if not additional_columns:
            return

        self.recorded_items.extend(additional_columns)
        csv_processor.expand_csv_columns(self._result_csv_file(), additional_columns)

    def on_weights_received(self, server, weights_received):
        """
        Event called after the updated weights have been received.
        """
        logging.info("[%s] Updated weights have been received.", server)

    def on_weights_aggregated(self, server, updates):
        """
        Event called after the updated weights have been aggregated.
        """
        logging.info("[%s] Finished aggregating updated weights.", server)

    def on_clients_selected(self, server, selected_clients, **kwargs):
        """
        Event called after clients have been selected in each round.
        """

    def on_clients_processed(self, server, **kwargs):
        """Additional work to be performed after client reports have been processed."""
        # Record results into a .csv file
        logged_items = server.get_logged_items()
        self._ensure_evaluation_columns(logged_items)
        new_row = []
        for item in self.recorded_items:
            item_value = logged_items.get(item)
            new_row.append(item_value)

        csv_processor.write_csv(self._result_csv_file(), new_row)

        if (
            hasattr(Config().clients, "do_test")
            and Config().clients.do_test
            and (
                hasattr(Config(), "results")
                and hasattr(Config().results, "record_clients_accuracy")
                and Config().results.record_clients_accuracy
            )
        ):
            # Updates the log for client test accuracies
            accuracy_csv_file = (
                f"{Config().params['result_path']}/{os.getpid()}_accuracy.csv"
            )

            for update in server.updates:
                accuracy_row = [
                    server.current_round,
                    update.client_id,
                    update.report.accuracy,
                ]
                csv_processor.write_csv(accuracy_csv_file, accuracy_row)

        logging.info("[%s] All client reports have been processed.", server)

    def on_training_will_start(self, server, **kwargs):
        """
        Event called before selecting clients for the first round of training.
        """
        logging.info("[%s] Starting training.", server)

    def on_server_will_close(self, server, **kwargs):
        """
        Event called at the start of closing the server.
        """
        logging.info("[%s] Closing the server.", server)
