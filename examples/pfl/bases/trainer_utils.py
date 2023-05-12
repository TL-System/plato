"""
The necessary tools used by trainers.
"""

import os

import pandas as pd


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


def checkpoint_personalized_accuracy(
    result_path, client_id, accuracy, current_round, epoch, run_id
):
    """Save the personaliation accuracy to the results dir."""

    save_location = os.path.join(result_path, "client_" + str(client_id))

    filename = NameFormatter.get_format_name(
        client_id=client_id, suffix="personalized_accuracy", ext="csv"
    )
    os.makedirs(save_location, exist_ok=True)
    save_personalized_accuracy(
        accuracy,
        current_round=current_round,
        epoch=epoch,
        accuracy_type="personalized_accuracy",
        filename=filename,
        location=save_location,
    )
