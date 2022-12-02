"""
Save and load desired checkpoints.
"""
import os
import re
from typing import Dict, Optional, List

import torch

from plato.utils.filename_formatter import get_format_name


class CheckpointsOperator:
    """The saver for checkpoints that includes pretrained models and checkpoints models.

    We call it CheckpointsSaver as the pretrained models can be regarded as one type
    of checkpoint.
    """

    def __init__(self, checkpoints_dir="checkpoints/"):
        """Initialize the loader with the directory where checkpoints should be stored."""
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        check_points_name: str,
        optimizer_state_dict: dict,
        lr_scheduler_state_dict: dict,
        epoch: int,
        config_args: dict,
    ) -> bool:
        # pylint:disable=too-many-arguments
        """Save the checkpoint to specific dir.

        :param model_state_dict: A Dict holding the state of a to-be-saved model
        :param check_points_name: The string for the name of the checkpoint file.
        :param optimizer_state_dict: A Dict holding the state of a to-be-saved optimizer.
        :param lr_scheduler_state_dict: A Dict holding the state of a to-be-saved lr_scheduler.
        :param epoch (int): A Integer presenting the epoch number.
        :param config_args (dict): A Dict containing the hyper-parameters
        """
        checkpoint_paths = [
            os.path.join(self.checkpoints_dir, checkpoint_name)
            for checkpoint_name in check_points_name
        ]

        for checkpoint_path in checkpoint_paths:
            torch.save(
                {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                    "lr_scheduler": lr_scheduler_state_dict,
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
        search_words: List[str],
        anchor_metric: str = "round",
        mask_anchors: Optional[List[str]] = None,
    ):
        """Search the latest checkpoint file under the checkpoint dir based on the key_words.
            The 'anchor_metric' is utilized to measure to 'latest'
            The 'mask_anchors' term is utilized to filter out unrelated terms.

        :param key_words: A list holding the key words contained in the file name.
        :param anchor_metric: A string presenting the metric used to measure the latest.
        :param mask_anchors: A list holding strings that should be ignored when searching
            for the file name.
        """

        if mask_anchors is None:
            mask_anchors = ["epohs"]

        def is_masked_file(ckp_file):
            return any(anchor in ckp_file for anchor in mask_anchors)

        def is_required_file(ckp_file):
            return all(word in ckp_file for word in search_words if word is not None)

        checkpoint_files = [
            ckp_file
            for ckp_file in os.listdir(self.checkpoints_dir)
            if not is_masked_file(ckp_file) and is_required_file(ckp_file)
        ]

        latest_checkpoint_filename = None
        latest_number = 0
        for ckp_file in checkpoint_files:
            obtained_anchor = re.findall(f"{anchor_metric}", ckp_file)[0]
            anchor_value = int(re.findall(r"\d+", obtained_anchor)[0])
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


def get_client_checkpoint_operator(
    client_id: int, target_checkpoint_dir: str, current_round: int
):
    """Get checkpoint operator specific for clients."""

    client_cpk_dir = os.path.join(target_checkpoint_dir, "client_" + str(client_id))
    cpk_oper = CheckpointsOperator(checkpoints_dir=client_cpk_dir)
    return cpk_oper


def perform_client_checkpoint_saving(
    client_id: int,
    model_name: str,
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: dict,
    lr_schedule_state_dict: dict,
    config: dict,
    present_epoch: int,
    base_epoch: int,
    prefix: Optional[str] = None,
):
    # pylint:disable=too-many-arguments

    """Save the checkpoint for sepcific client."""
    current_round = config["current_round"]
    # run_id = config['run_id']
    # we have to set the run_id to be None here as the client can
    # have different run id in the whole training process.
    run_id = None
    cpk_oper = get_client_checkpoint_operator(client_id, current_round)

    # Before the training, we expect to save the initial
    # model of this round
    filename = get_format_name(
        model_name=model_name,
        client_id=client_id,
        round_n=current_round,
        epoch_n=present_epoch,
        run_id=run_id,
        prefix=prefix,
        ext="pth",
    )
    cpk_oper.save_checkpoint(
        model_state_dict=model_state_dict,
        check_points_name=[filename],
        optimizer_state_dict=optimizer_state_dict,
        lr_scheduler_state_dict=lr_schedule_state_dict,
        epoch=base_epoch,
        config_args=config,
    )

    return filename


def perform_client_checkpoint_loading(
    client_id: int,
    model_name: str,
    current_round: int,
    run_id: int,
    epoch: int,
    prefix: Optional[str] = None,
    anchor_metric: str = "round",
    mask_anchors: Optional[List[str]] = None,
    use_latest: bool = True,
):
    # pylint:disable=too-many-arguments

    """Performing checkpoint loading.

    :param use_latest: A boolean to show whether utilize the latest checkpoint file
        if the required file does not exist.
    """
    if mask_anchors is None:
        mask_anchors = ["epohs"]

    cpk_oper = get_client_checkpoint_operator(client_id, current_round)

    # Before the training, we expect to save the initial
    # model of this round
    filename = get_format_name(
        model_name=model_name,
        client_id=client_id,
        round_n=current_round,
        epoch_n=epoch,
        run_id=run_id,
        prefix=prefix,
        ext="pth",
    )

    if use_latest:
        if not cpk_oper.vaild_checkpoint_file(filename):
            # Loading the latest checkpoint file
            key_words = [model_name, prefix]
            filename = cpk_oper.search_latest_checkpoint_file(
                key_words=key_words,
                anchor_metric=anchor_metric,
                mask_anchors=mask_anchors,
            )

    return filename, cpk_oper

