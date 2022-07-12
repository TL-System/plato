import os
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

    def invaild_checkpoint_file(self, filename):
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
    run_id = config['run_id']
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
                                      prefix=None):

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

    return filename, cpk_oper
