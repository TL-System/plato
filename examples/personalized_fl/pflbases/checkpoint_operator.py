"""
The implementation of checkpoints operations, including
 - checkpoint saving
 - checkpoint loading
 - checkpoint searching
"""
import os
import re
from typing import Dict, Optional, List

import torch

from plato.utils.filename_formatter import NameFormatter


class CheckpointsOperator:
    """A base operator for checkpoint operations."""

    def __init__(self, checkpoints_dir: str = "checkpoints/"):
        """Initialize the directory where checkpoints should be stored or loaded."""
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    # pylint:disable=too-many-arguments
    def save_checkpoint(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        checkpoints_name: List[str],
        optimizer_state_dict: Optional[dict] = None,
        lr_scheduler_state_dict: Optional[dict] = None,
        learning_dict: Optional[dict] = None,
        epoch: Optional[int] = None,
        config_args: Optional[dict] = None,
    ) -> bool:
        """Save the checkpoint to specific dir.

        :param model_state_dict: A Dict holding the state of a to-be-saved model.
        :param checkpoints_name: The List contains strings for the names of checkpoint
         files. This supports saving the checkpoint to multiple pieces, each corresponding
         to one string name.
        :param optimizer_state_dict: A Dict holding the state of a to-be-saved optimizer.
         Default to be None for not saving.
        :param lr_scheduler_state_dict: A Dict holding the state of a to-be-saved lr_scheduler.
         Default to be None for not saving.
        :param learning_dict: A Dict holding the state of the learning process. It can
         include "loss" for example.
         Default to be None for not saving.
        :param epoch: An Integer presenting the epoch number.
         Default to be None for not saving.
        :param config_args: A Dict containing the hyper-parameters.
         Default to be None for not saving.
        """
        checkpoint_paths = [
            os.path.join(self.checkpoints_dir, ckpt_name)
            for ckpt_name in checkpoints_name
        ]

        for checkpoint_path in checkpoint_paths:
            torch.save(
                {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                    "lr_scheduler": lr_scheduler_state_dict,
                    "learning": learning_dict,
                    "epoch": epoch,
                    "args": config_args,
                },
                checkpoint_path,
            )

        return True

    def load_checkpoint(self, checkpoint_name: str):
        """Load the checkpoint."""
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)

        return torch.load(checkpoint_path, map_location=torch.device("cpu"))

    def search_latest_checkpoint_file(
        self,
        search_key_words: List[str],
        anchor_metric: str = "round",
        filter_words: Optional[List[str]] = None,
    ):
        """Search the latest checkpoint file under the checkpoint dir based on 'search_key_words'.
        The 'anchor_metric' is utilized to measure what is the latest.
        The 'filter_words' term is utilized to filter out unrelated files.
        """

        if filter_words is None:
            filter_words = ["epochs"]

        def is_filterd_file(ckp_file):
            return any(word in ckp_file for word in filter_words)

        def is_required_file(ckp_file):
            return all(
                word in ckp_file for word in search_key_words if word is not None
            )

        checkpoint_files = [
            ckp_file
            for ckp_file in os.listdir(self.checkpoints_dir)
            if not is_filterd_file(ckp_file)
            and is_required_file(ckp_file)
            and re.search(r"\.pth$", ckp_file)
        ]

        latest_checkpoint_filename = None
        latest_number = 0
        for ckp_file in checkpoint_files:
            pattern = re.escape(anchor_metric) + r"(\d+)"
            obtained_anchor = re.search(pattern, ckp_file, re.IGNORECASE)
            anchor_value = int(obtained_anchor.group(1))
            if anchor_value >= latest_number:
                latest_number = anchor_value
                latest_checkpoint_filename = ckp_file

        return latest_checkpoint_filename

    def vaild_checkpoint_file(self, filename: str):
        """Check whether the file exists."""
        file_path = os.path.join(self.checkpoints_dir, filename)
        if os.path.exists(file_path):
            return True
        return False


# pylint:disable=too-many-arguments
def save_client_checkpoint(
    client_id: int,
    model_name: str,
    checkpoints_dir: str,
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Optional[dict] = None,
    lr_scheduler_state_dict: Optional[dict] = None,
    learning_dict: Optional[dict] = None,
    config: Optional[dict] = None,
    global_epoch: Optional[int] = None,
    local_epoch: Optional[int] = None,
    prefix: Optional[str] = None,
) -> str:
    """Save the checkpoint for a specific client."""
    config = config if config is not None else {}
    current_round = config["current_round"] if "current_round" in config else None
    # run_id = config['run_id']
    # we have to set the run_id to be None here as the client can
    # have different run id in the whole training process.
    run_id = None

    cpk_oper = CheckpointsOperator(checkpoints_dir=checkpoints_dir)

    # Before the training, we expect to save the initial
    # model of this round
    filename = NameFormatter.get_format_name(
        model_name=model_name,
        client_id=client_id,
        round_n=current_round,
        epoch_n=local_epoch,
        run_id=run_id,
        prefix=prefix,
        ext="pth",
    )
    cpk_oper.save_checkpoint(
        model_state_dict=model_state_dict,
        checkpoints_name=[filename],
        optimizer_state_dict=optimizer_state_dict,
        lr_scheduler_state_dict=lr_scheduler_state_dict,
        learning_dict=learning_dict,
        epoch=global_epoch,
        config_args=config,
    )

    return filename


# pylint:disable=too-many-arguments
def search_client_checkpoint(
    client_id: int,
    model_name: str,
    checkpoints_dir: str,
    current_round: Optional[int] = None,
    run_id: Optional[int] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
    anchor_metric: str = "round",
    mask_words: Optional[List[str]] = None,
    use_latest: bool = True,
) -> CheckpointsOperator:
    """Search for the target checkpoint of the client.

    :param use_latest: A boolean to show whether to utilize the
     latest checkpoint file if the required file does not exist. .
    """
    if mask_words is None:
        mask_words = ["epoch"]

    cpk_oper = CheckpointsOperator(checkpoints_dir=checkpoints_dir)

    # Before the training, we expect to save the initial
    # model of this round
    filename = NameFormatter.get_format_name(
        model_name=model_name,
        client_id=client_id,
        round_n=current_round,
        epoch_n=epoch,
        run_id=run_id,
        prefix=prefix,
        ext="pth",
    )

    if cpk_oper.vaild_checkpoint_file(filename):
        return filename, True
    else:
        if use_latest:
            # Loading the latest checkpoint file
            search_key_words = [model_name, prefix]
            filename = cpk_oper.search_latest_checkpoint_file(
                search_key_words=search_key_words,
                anchor_metric=anchor_metric,
                filter_words=mask_words,
            )
            return filename, True

    return filename, False
