import os
import torch
import logging


class CheckpointsSaver(object):
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
