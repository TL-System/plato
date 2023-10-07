"""
The necessary tools used by trainers.
"""

import os
import logging

import pandas as pd
import numpy as np

import torch

from pflbases import fedavg_partial

from plato.utils.filename_formatter import NameFormatter


def save_personalized_accuracy(
    accuracy,
    current_round=None,
    epoch=None,
    accuracy_type="test_accuracy",
    filename=None,
    location=None,
):
    # pylint:disable=too-many-arguments
    """Saving the test accuracy to a file."""

    to_save_path = os.path.join(location, filename)
    current_round = current_round if current_round is not None else 0
    current_epoch = epoch if epoch is not None else 0
    acc_dataframe = pd.DataFrame(
        {"round": current_round, "epoch": current_epoch, accuracy_type: accuracy},
        index=[0],
    )

    is_use_header = not os.path.exists(to_save_path)
    acc_dataframe.to_csv(to_save_path, index=False, mode="a", header=is_use_header)


def load_personalized_accuracy(
    current_round=None,
    accuracy_type="test_accuracy",
    filename=None,
    location=None,
):
    """Loading the test accuracy from a file."""

    load_path = os.path.join(location, filename)
    loaded_rounds_accuracy = pd.read_csv(load_path)
    if current_round is None:
        # default use the last row
        desired_row = loaded_rounds_accuracy.iloc[-1]
    else:
        desired_row = loaded_rounds_accuracy.loc[
            loaded_rounds_accuracy["round"] == current_round
        ]
        desired_row = loaded_rounds_accuracy.iloc[-1]

    accuracy = desired_row[accuracy_type]

    return accuracy


def checkpoint_personalized_metrics(
    result_path, client_id, metrics_holder, current_round, epoch, run_id
):
    """Save the personaliation accuracy to the results dir."""

    save_location = os.path.join(result_path, "client_" + str(client_id))

    accuracy = metrics_holder.accuracy

    acc_filename = NameFormatter.get_format_name(
        client_id=client_id, suffix="personalized_accuracy", ext="csv"
    )
    labels_filename = NameFormatter.get_format_name(
        client_id=client_id,
        round_n=current_round,
        epoch_n=epoch,
        suffix="personalized_labels",
        ext="npy",
    )
    predictions_filename = NameFormatter.get_format_name(
        client_id=client_id,
        round_n=current_round,
        epoch_n=epoch,
        suffix="personalized_predictions",
        ext="npy",
    )
    encodings_filename = NameFormatter.get_format_name(
        client_id=client_id,
        round_n=current_round,
        epoch_n=epoch,
        suffix="personalized_encodings",
        ext="npy",
    )
    os.makedirs(save_location, exist_ok=True)
    save_personalized_accuracy(
        accuracy,
        current_round=current_round,
        epoch=epoch,
        accuracy_type="personalized_accuracy",
        filename=acc_filename,
        location=save_location,
    )
    np.save(os.path.join(save_location, labels_filename), metrics_holder.labels)
    np.save(
        os.path.join(save_location, predictions_filename), metrics_holder.predictions
    )

    if metrics_holder.encodings is not None:
        np.save(
            os.path.join(save_location, encodings_filename), metrics_holder.encodings
        )


class MetircsCollector:
    """A collector holding the necessary metrics."""

    def __init__(self) -> None:
        self.accuracy = -1
        self.labels = None
        self.predictions = None
        self.encodings = None

    def reset(self):
        """Resetting items"""
        self.accuracy = -1
        self.labels = None
        self.predictions = None
        self.encodings = None

    def set_accuracy(self, accuracy):
        """Setting the accuracy"""
        self.accuracy = accuracy

    def add_labels(self, labels):
        """Adding one batch of labels."""
        self.labels = (
            labels if self.labels is None else torch.cat((self.labels, labels), dim=0)
        )

    def add_predictions(self, predictions):
        """Adding one batch of predictions."""
        self.predictions = (
            predictions
            if self.predictions is None
            else torch.cat((self.predictions, predictions), dim=0)
        )

    def add_encodings(self, encodings):
        """Adding one batch samples of encodings."""
        self.encodings = (
            encodings
            if self.encodings is None
            else torch.cat((self.encodings, encodings), dim=0)
        )


def freeze_model(model, modules_name=None, log_info: str = ""):
    """Freezing a part of the model."""
    if modules_name is not None:
        frozen_params = []
        for name, param in model.named_parameters():
            if any([param_name in name for param_name in modules_name]):
                param.requires_grad = False
                frozen_params.append(name)

        logging.info(
            "%s has frozen %s",
            log_info,
            fedavg_partial.Algorithm.extract_modules_name(frozen_params),
        )


def activate_model(model, modules_name=None):
    """Defreezing a part of the model."""
    if modules_name is not None:
        for name, param in model.named_parameters():
            if any([param_name in name for param_name in modules_name]):
                param.requires_grad = True
