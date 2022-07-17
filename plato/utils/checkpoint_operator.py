import os
import re

import torch

from plato.utils.arrange_saving_name import get_format_name
from plato.config import Config


class CheckpointsOperator(object):
    """ The saver for checkpoints that includes pretrained models and checkpoints models.

        We call it CheckpointsSaver as the pretrained models can be regarded as one type
            of checkpoint.
     """

    def __init__(self, checkpoints_dir='checkpoints/'):
        """ Initialize the loader with the directory where checkpoints should be stored """
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_checkpoint(self, model_state_dict, check_points_name,
                        optimizer_state_dict, lr_scheduler_state_dict, epoch,
                        config_args):
        """ Save the checkpoint to specific dir.

            Args:
                model (torch model): Model to be saved
                check_points_name (str): name for the checkpoint file.
                optimizer (torch.optimizer): a optimizer object.
                lr_scheduler (torch.lr_scheduler): a lr_scheduler object.
                epoch (int): epoch nuber.
                config_args (dict): the config parameters to be saved
        """
        checkpoint_paths = [
            os.path.join(self.checkpoints_dir, checkpoint_name)
            for checkpoint_name in check_points_name
        ]

        for checkpoint_path in checkpoint_paths:
            torch.save(
                {
                    'model': model_state_dict,
                    'optimizer': optimizer_state_dict,
                    'lr_scheduler': lr_scheduler_state_dict,
                    'epoch': epoch,
                    'args': config_args,
                }, checkpoint_path)

        return True

    def load_checkpoint(self, checkpoint_name):
        """ Load the checkpoint to specific dir.

            Args:
                model (torch model): Model to be saved
                check_points_name (str): name for the checkpoint file.
                optimizer (torch.optimizer): a optimizer object.
                lr_scheduler (torch.lr_scheduler): a lr_scheduler object.
                epoch (int): epoch nuber.
                config_args (dict): the config parameters to be saved
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)

        return torch.load(checkpoint_path)

    def search_latest_checkpoint_file(self,
                                      anchor_metric="round",
                                      mask_anchors="epoch"):
        """ Search the latest checkpoint file under the checkpoint dir. """
        checkpoint_files = [
            ckp_file for ckp_file in os.listdir(self.checkpoints_dir)
            if all([anchor not in ckp_file for anchor in mask_anchors])
        ]

        latest_checkpoint_filename = None
        latest_number = 0
        for ckp_file in checkpoint_files:
            obtained_anchor = re.findall(r'%s\d+' % anchor_metric, ckp_file)[0]
            anchor_value = int(re.findall(r'\d+', obtained_anchor)[0])
            if anchor_value >= latest_number:
                latest_number = anchor_value
                latest_checkpoint_filename = ckp_file

        return latest_checkpoint_filename

    def vaild_checkpoint_file(self, filename):
        file_path = os.path.join(self.checkpoints_dir, filename)
        if os.path.exists(file_path):
            return True
        else:
            return False


def get_client_checkpoint_operator(client_id, current_round):
    if current_round == Config().trainer.rounds:
        target_dir = Config().params['model_path']
    else:
        target_dir = Config().params['checkpoint_path']

    client_cpk_dir = os.path.join(target_dir, "client_" + str(client_id))
    cpk_oper = CheckpointsOperator(checkpoints_dir=client_cpk_dir)
    return cpk_oper


def perform_client_checkpoint_saving(client_id,
                                     model_name,
                                     model_state_dict,
                                     optimizer_state_dict,
                                     lr_schedule_state_dict,
                                     config,
                                     kwargs,
                                     present_epoch,
                                     base_epoch,
                                     prefix=None):

    current_round = kwargs['current_round']
    # run_id = config['run_id']
    # we have to set the run_id to be None here as the client can
    # have different run id in the whole training process.
    run_id = None
    cpk_oper = get_client_checkpoint_operator(client_id, current_round)

    # Before the training, we expect to save the initial
    # model of this round
    filename = get_format_name(model_name=model_name,
                               client_id=client_id,
                               round_n=current_round,
                               epoch_n=present_epoch,
                               run_id=run_id,
                               prefix=prefix,
                               ext="pth")
    cpk_oper.save_checkpoint(model_state_dict=model_state_dict,
                             check_points_name=[filename],
                             optimizer_state_dict=optimizer_state_dict,
                             lr_scheduler_state_dict=lr_schedule_state_dict,
                             epoch=base_epoch,
                             config_args=Config().to_dict())

    return filename


def perform_client_checkpoint_loading(client_id,
                                      model_name,
                                      current_round,
                                      run_id,
                                      epoch,
                                      prefix=None,
                                      anchor_metric="round",
                                      mask_anchors=["epoch"],
                                      use_latest=True):
    """ Performing checkpoint loading.

        use_latest: Using the latest checkpoint file if the required file does not
                    exist.
    """
    cpk_oper = get_client_checkpoint_operator(client_id, current_round)

    # Before the training, we expect to save the initial
    # model of this round
    filename = get_format_name(model_name=model_name,
                               client_id=client_id,
                               round_n=current_round,
                               epoch_n=epoch,
                               run_id=run_id,
                               prefix=prefix,
                               ext="pth")
    if use_latest:
        if not cpk_oper.vaild_checkpoint_file(filename):
            # Loading the latest checkpoint file
            filename = cpk_oper.search_latest_checkpoint_file(
                anchor_metric=anchor_metric, mask_anchors=mask_anchors)

    return filename, cpk_oper
