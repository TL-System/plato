"""
The trainer defined for the adaptive Adaptive hierarchical gradient blending method

"""

import os
import logging
import multiprocessing as mp

import numpy as np
import torch

import wandb
from plato.config import Config
from plato.utils import optimizers
from plato.trainers import basic

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

        self.init_trajectory_items()

    def init_trajectory_items(self):
        """ Initialize the containers to hold the train trajectory"""
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

    def backtrack_gradient_trajectory(self, trajectory_idx):
        """ Record the gradient """
        assert trajectory_idx < len(self.gradients_trajectory)

        return self.gradients_trajectory[trajectory_idx]

    def backtrack_loss_trajectory(self, mode, modality_name, trajectory_idx):
        """ Record the loss for the required modality """
        assert mode in list(self.losses_trajectory.keys())
        mode_trajs = self.losses_trajectory[mode]
        assert modality_name in list(mode_trajs.keys())
        mode_mdl_trajs = self.losses_trajectory[mode][modality_name]

        assert trajectory_idx < len(mode_mdl_trajs)
        return mode_mdl_trajs[trajectory_idx]

    def backtrack_loss_trajectories(self, mode, modality_name,
                                    trajectory_idxs):
        """ Record the modality multi-steps loss """
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
        """ Record multiple modalities' losses in the step trajectory_idx """
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
            backtracked_multimodal_mode_losses[mdl_name] = mode_trajs[
                mdl_name][trajectory_idx]

        return backtracked_multimodal_mode_losses

    @torch.no_grad()
    def eval_step(self, eval_data_loader, num_iters=None, model=None):
        """ Perfome the evaluation """
        if model is None:
            model = self.model

        model.eval()

        eval_avg_losses = dict()

        for batch_id, (examples, labels) in enumerate(eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)

            losses = model(data_container=examples,
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

    def obtain_local_global_ogr_items(self, trainset, evalset):
        """ We can directly call the self.model in this function to get the global model
            because the weights from the server are assigned to the client before training """

        # we can call eval directly to get the performance of the global model on the local dataset
        # prepare data loaders
        eval_loader = torch.utils.data.DataLoader(dataset=evalset,
                                                  shuffle=False,
                                                  batch_size=1,
                                                  num_workers=Config.data.get(
                                                      'workers_per_gpu', 1))
        # 1. obtain the eval loss of the received global model
        eval_avg_losses = self.eval_step(eval_data_loader=eval_loader)

        # obtain the training loss of the received global model
        eval_trainset_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=1,
            num_workers=Config.data.get('workers_per_gpu', 1))

        # get the averaged loss on 50 batch size
        eval_subtrainset_avg_losses = self.eval_step(
            eval_data_loader=eval_trainset_loader, num_iters=50)

        # 2. extract the eval and train loss of the local model
        #   this part of value should be stored in the last position of the loss trajectory

        local_train_avg_losses = self.backtrack_multimodal_loss_trajectory(
            mode="train",
            modality_names=["RGB", "Flow", "Audio", "Fused"],
            trajectory_idx=-1)
        local_eval_avg_losses = self.backtrack_multimodal_loss_trajectory(
            mode="eval",
            modality_names=["RGB", "Flow", "Audio", "Fused"],
            trajectory_idx=-1)

        return eval_avg_losses, eval_subtrainset_avg_losses, \
                local_eval_avg_losses, local_train_avg_losses

    def reweight_losses(self, blending_weights, losses):
        """[Reweight the losses to achieve the gradient blending]

        Args:
            blending_weights ([dict]): contains the blending weight of each modality network
                                        {"RGB": float, "Flow": float}
            losses ([dict]): contains the loss of each modality network
                                        {"RGB": float, "Flow": float}
        """
        modality_names = list(blending_weights.keys())
        reweighted_losses = dict()
        for modl_nm in modality_names:
            reweighted_losses[
                modl_nm] = blending_weights[modl_nm] * losses[modl_nm]

        return reweighted_losses

    def train_process(self, config, trainset, evalset, sampler,
                      blending_weights):
        log_interval = config.log_config["interval"]
        batch_size = config.trainer['batch_size']

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        # prepare traindata loaders
        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   sampler=sampler.get(),
                                                   num_workers=config.data.get(
                                                       'workers_per_gpu', 1))

        eval_loader = torch.utils.data.DataLoader(dataset=evalset,
                                                  shuffle=False,
                                                  batch_size=batch_size,
                                                  sampler=sampler.get(),
                                                  num_workers=config.data.get(
                                                      'workers_per_gpu', 1))

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()
        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer",
                                optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)
        # Initializing the learning rate schedule, if necessary
        if hasattr(config, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        # operate the local training
        supported_modalities = trainset.supported_modalities
        # in order to blend the gradients in the server side
        #   The eval/train loss of the first and last epoches should be recorded
        for epoch in range(1, epochs + 1):
            epoch_train_losses = {
                modl_nm: 0.0
                for modl_nm in supported_modalities
            }
            total_batches = 0
            total_epoch_loss = 0
            for batch_id, (multimodal_examples,
                           labels) in enumerate(train_loader):
                labels = labels.to(self.device)

                optimizer.zero_grad()

                losses = self.model.forward(data_container=multimodal_examples,
                                            label=labels,
                                            return_loss=True)

                weighted_losses = self.reweight_losses(blending_weights,
                                                       losses)

                # added the losses
                weighted_global_loss = 0
                for modl_nm in supported_modalities:
                    epoch_train_losses[modl_nm] += weighted_losses[modl_nm]
                    weighted_global_loss += weighted_losses[modl_nm]

                total_epoch_loss += weighted_global_loss

                weighted_global_loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_schedule.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                            format(os.getpid(), epoch, epochs, batch_id,
                                   len(train_loader),
                                   weighted_losses.data.item()))
                    else:
                        if hasattr(config, 'use_wandb'):
                            wandb.log(
                                {"batch loss": weighted_losses.data.item()})

                        logging.info(
                            "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                            format(self.client_id, epoch, epochs, batch_id,
                                   len(train_loader),
                                   weighted_losses.data.item()))
                total_batches = batch_id
            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # only record the first and final performance of the local epoches
            if epoch == 1 or epoch == epochs:
                epoch_avg_train_loss = total_epoch_loss / (total_batches + 1)

                eval_avg_losses = self.eval_step(eval_data_loader=eval_loader)
                weighted_eval_losses = self.reweight_losses(
                    blending_weights, eval_avg_losses)
                total_eval_loss = 0
                for modl_nm in supported_modalities:
                    modl_train_avg_loss = epoch_train_losses[
                        modl_nm] / total_batches
                    modl_eval_avg_loss = eval_avg_losses[modl_nm]
                    if modl_nm not in list(
                            self.mm_train_losses_trajectory.keys()):
                        self.mm_train_losses_trajectory[
                            modl_nm] = modl_train_avg_loss
                    else:
                        self.mm_train_losses_trajectory[modl_nm].append(
                            modl_train_avg_loss)
                    if modl_nm not in list(
                            self.mm_val_losses_trajectory.keys()):
                        self.mm_val_losses_trajectory[
                            modl_nm] = modl_eval_avg_loss
                    else:
                        self.mm_val_losses_trajectory[modl_nm].append(
                            modl_eval_avg_loss)

                    total_eval_loss += weighted_eval_losses[modl_nm]

                # store the global losses
                self.global_mm_train_losses_trajectory.append(
                    epoch_avg_train_loss)
                self.global_mm_val_losses_trajectory.append(total_eval_loss)
        self.model.cpu()

        model_type = config['model_name']
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
        self.save_model(filename)

        if 'use_wandb' in config:

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)
        if 'use_wandb' in config:
            run.finish()

    def train(self, trainset, evalset, sampler, blending_weights) -> bool:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.

        Returns:
            Whether training was successfully completed.
        """
        self.start_training()

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        train_proc = mp.Process(target=Trainer.train_process,
                                args=(
                                    self,
                                    config,
                                    trainset,
                                    evalset,
                                    sampler,
                                    blending_weights,
                                ))
        train_proc.start()
        train_proc.join()

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"
        try:
            self.load_model(filename)
        except OSError:  # the model file is not found, training failed
            logging.info("The training process on client #%d failed.",
                         self.client_id)
            self.run_sql_statement("DELETE FROM trainers WHERE run_id = (?)",
                                   (self.client_id, ))
            return False

        self.pause_training()
        return True
