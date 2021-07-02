#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from plato.config import Config
from plato.trainers import basic
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# basic.Trainer
#   arguments: model=None
#       One can either set the model in the initialization or the trainer will
#   define these itself based on the configuration file

#   # The functions are required by the client
#   - zeros: server will call this function to generate the zeros tensor
#   - save_model: Saving the model to a file
#   - load_model: Loading pre-trained model weights from a file.
#   - train_process(config, trainset, sampler): the actual training process that get the dataloader
#       from the trainset and operate the full local training epoches
#       * train_model, you can implement a new train_model func to achieve the training
#   - train(trainset, sampler): operate the train_process in a sync manner


class Trainer(basic.Trainer):
    """The federated learning trainer for the adaptive gradient blending client. """
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """

        super().__init__(model)

        self.is_distributed = Config().is_parallel()

        # mm is the abbreviation of multimodal 

        # For the Overfitting:
        # record the training losse:
        #  each item of the dict is a list containing the losses of the specific modality
        self.mm_train_losses_trajectory = dict()
        # the global training losses - sum of all losses
        self.global_mm_train_losses_trajectory = list()

        # For the Generalization:
        # record the val losse:
        #  each item of the dict is a list containing the losses of the specific modality
        self.mm_val_losses_trajectory = dict()
        # the global val losses - sum of all losses
        self.global_mm_val_losses_trajectory = list()

        self.losses_trajectory = {
            "train": self.mm_train_losses_trajectory,
            "val": self.mm_val_losses_trajectory
        }

        self.global_losses_trajectory = {
            "train": self.global_mm_train_losses_trajectory,
            "val": self.global_mm_val_losses_trajectory
        }

        # 
        self.mm_weights_trajectory = dict()
        self.mm_gradients_trajectory = dict()

    def backtrack_gradient_trajectory(self, trajectory_idx):
        assert gradient_idx < len(self.gradients_trajectory)

        return self.gradients_trajectory[trajectory_idx]

    def backtrack_loss_trajectory(self, mode, modality_name, trajectory_idx):

        assert mode in list(self.losses_trajectory.keys())
        mode_trajs = self.losses_trajectory[mode]
        assert modality_name in list(mode_trajs.keys())
        mode_mdl_trajs = self.losses_trajectory[mode][modality_name]

        assert trajectory_idx < len(mode_mdl_trajs)
        return mode_mdl_trajs[trajectory_idx]

    def backtrack_loss_trajectories(self, mode, modality_name,
                                    trajectory_idxs):
        assert mode in list(self.losses_trajectory.keys())
        mode_trajs = self.losses_trajectory[mode]
        assert modality_name in list(mode_trajs.keys())
        mode_mdl_trajs = self.losses_trajectory[mode][modality_name]

        assert all(
            [traj_idx < len(mode_mdl_trajs) for traj_idx in trajectory_idxs])

        backtracked_mode_mdl_losses = dict()
        backtracked_mode_mdl_losses[modality_name] = [
            mode_mdl_trajs[traj_idx] for traj_idx in trajectory_idxs
        ]
        return backtracked_mode_mdl_losses

    def backtrack_multimodal_loss_trajectory(self, mode, modality_names,
                                             trajectory_idx):
        assert mode in list(self.losses_trajectory.keys())
        mode_trajs = self.losses_trajectory[mode]

        assert all([
            mdl_name in list(mode_trajs.keys()) for mdl_name in modality_names
        ])
        assert all([
            trajectory_idx < len(mode_trajs[mdl_name])
            for mdl_name in modality_names
        ])

        backtracked_multimodal_mode_losses = dict()
        for mdl_name in modality_names:
             backtracked_multimodal_mode_losses[mdl_name] = mode_trajs[mdl_name][trajectory_idx]

        return backtracked_multimodal_mode_losses

    @torch.no_grad()
    def eval_step(self, eval_data_loader, num_iters=None, model=None):
        if model == None:
            model = self.model

        model.eval()
        mode = 'val'
        eval_avg_losses = dict()
        eval_data_loader = eval_data_loader
        for batch_id, (examples,
                        labels) in enumerate(eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(
                self.device)
            
            losses = model(rgb_imgs=examples["RGB"],
                flow_imgs = examples["Flow"],
                audio_features = examples["Audio"],
                label=labels,
                return_loss=True)
            for loss_key in list(losses.keys()):
                if loss_key in list(eval_avg_losses.keys()):
                    eval_avg_losses[loss_key].append(losses[loss_key])
                else:
                    eval_avg_losses[loss_key] = [losses[loss_key]]

            if isinstance(num_iters, int) and batch_id > num_iters:
                break

        for loss_key in list(eval_avg_losses.keys()):
            eval_key_losses = eval_avg_losses[loss_key]
            eval_avg_losses[loss_key] = np.mean(np.array(eval_key_losses))

        return eval_avg_losses

    def obtain_local_global_OGR_items(self, trainset, evalset):
        """ We can directly call the self.model in this function to get the global model
            because the weights from the server are assigned to the client before training """
        
        # we can call eval directly to get the performance of the global model on the local dataset
        # prepare data loaders
        eval_loader = torch.utils.data.DataLoader(dataset=evalset,
                                                   shuffle=False,
                                                   batch_size=1,
                                                   sampler=sampler.get()
                                                   num_workers=config.data.get('workers_per_gpu', 1))
        # 1. obtain the eval loss of the received global model
        eval_avg_losses = self.eval_step(eval_data_loader = eval_loader)
        
        # obtain the training loss of the received global model
        eval_trainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=1,
                                                   sampler=sampler.get()
                                                   num_workers=config.data.get('workers_per_gpu', 1))
        
        # get the averaged loss on 50 batch size                                           
        eval_subtrainset_avg_losses = self.eval_step(eval_data_loader = eval_trainset_loader, num_iters=50)


        # 2. extract the eval and train loss of the local model
        #   this part of value should be stored in the last position of the loss trajectory

        local_train_avg_losses = self.backtrack_multimodal_loss_trajectory(mode="train", 
                                            modality_names=["RGB", "Flow", "Audio", "Fused"],
                                             trajectory_idx=-1)
        local_eval_avg_losses = self.backtrack_multimodal_loss_trajectory(mode="eval", 
                                            modality_names=["RGB", "Flow", "Audio", "Fused"],
                                             trajectory_idx=-1)

        return eval_avg_losses, eval_subtrainset_avg_losses, local_train_avg_losses, local_eval_avg_losses


    def multimodal_gradient_blending(self, original_losses):
        """ The main code for the local multimodal gradient blending """
        pass

    def train_process(self, config, trainset, evalset, sampler):
        log_interval = config.log_config["interval"]
        batch_size = config.trainer['batch_size']

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        # prepare traindata loaders
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   sampler=sampler.get()
                                                   num_workers=config.data.get('workers_per_gpu', 1))

        
        
        # put model on gpus
        if self.is_distributed:
            logging.info("Using Data Parallelism.")
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = nn.DataParallel(model)

        # build runner
        optimizer = self.build_optimizer(model, config.optimizer)

        # conduct the local training epoches
