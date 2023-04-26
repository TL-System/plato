"""
The implementation of checkpoints operations, such as saving and loading.
"""
import os
import re
from typing import Dict, Optional, List

import torch

from plato.utils.filename_formatter import NameFormatter


class CheckpointsOperator:
    """The operations for checkpoints, including pretrained models and checkpoints models.

    This class is called CheckpointsOperator, as the pre-trained models can also
    be regarded as one type of checkpoint.
    """

    def __init__(self, checkpoints_dir: str = "checkpoints/"):
        """Initialize the directory where checkpoints should be stored or loaded.

        :param checkpoints_dir: A string to show where checkpoint operations perform.
        """
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

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
        # pylint:disable=too-many-arguments
        """Save the checkpoint to specific dir.

        :param model_state_dict: A Dict holding the state of a to-be-saved model
        :param checkpoints_name: The List containg strings for names of checkpoint files.
            This support saving the checkpoint to multiple pieces, each piece corresponding
            to one string name within 'checkpoints_name'.
        :param optimizer_state_dict: A Dict holding the state of a to-be-saved optimizer.
            Default to be None for not saving.
        :param lr_scheduler_state_dict: A Dict holding the state of a to-be-saved lr_scheduler.
            Default to be None for not saving.
        :param learning_dict: A Dict holding the state of the learning process. It can
            include "loss" for example. Default to be None for not saving.
        :param epoch: A Integer presenting the epoch number.
            Default to be None for not saving.
        :param config_args: A Dict containing the hyper-parameters
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

    def load_checkpoint(self, checkpoint_name):
        """Load the checkpoint to specific dir.

        :param checkpoint_name: The string for the name of the checkpoint file.
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)

        return torch.load(checkpoint_path)

    def search_latest_checkpoint_file(
        self,
        search_key_words: List[str],
        anchor_metric: str = "round",
        filter_words: Optional[List[str]] = None,
    ):
        """Search the latest checkpoint file under the checkpoint dir based on 'search_key_words'.
            The 'anchor_metric' is utilized to measure what is 'latest'.
            The 'filter_words' term is utilized to filter out unrelated files.

        :param search_key_words: A list holding the words for searching target files.
        :param anchor_metric: A string presenting the metric used to measure the latest.
        :param filter_words: A list holding strings that should be ignored when searching
            for the file name.
        """

        if filter_words is None:
            filter_words = ["epohs"]

        def is_filterd_file(ckp_file):
            return any(word in ckp_file for word in filter_words)

        def is_required_file(ckp_file):
            return all(
                word in ckp_file for word in search_key_words if word is not None
            )

        checkpoint_files = [
            ckp_file
            for ckp_file in os.listdir(self.checkpoints_dir)
            if not is_filterd_file(ckp_file) and is_required_file(ckp_file)
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
    # pylint:disable=too-many-arguments

    """Save the checkpoint for sepcific client.

    :param client_id: A integer to present the client id.
    :param model_name: A integer to present the model's name used
        for the checkpoint saving.
    :param checkpoints_dir: A string to present whether to save the
        checkpoints.
    :param model_state_dict: A Dict holding the state of a to-be-saved model
    :param optimizer_state_dict: A Dict holding the state of a to-be-saved optimizer.
        Default to be None for not saving.
    :param lr_scheduler_state_dict: A Dict holding the state of a to-be-saved lr_scheduler.
        Default to be None for not saving.
    :param learning_dict: A Dict holding the state of the learning process. It can
        include "loss" for example. Default to be None for not saving.
    :param global_epoch: A integer to present the client id.
    :param global_epoch: A integer to present global epoch.
    :param local_epoch: A integer to present local epoch within the client.
    :param prefix: A integer to present the client id.

    """
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


def load_client_checkpoint(
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
    # pylint:disable=too-many-arguments

    """Performing checkpoint loading.

    :param use_latest: A boolean to show whether utilize the latest checkpoint file
        if the required file does not exist.
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
        return filename, cpk_oper
    else:
        if use_latest:
            # Loading the latest checkpoint file
            search_key_words = [model_name, prefix]
            filename = cpk_oper.search_latest_checkpoint_file(
                search_key_words=search_key_words,
                anchor_metric=anchor_metric,
                filter_words=mask_words,
            )
            return filename, cpk_oper

    return filename, None
