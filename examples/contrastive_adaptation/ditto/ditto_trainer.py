"""
A personalized federated learning trainer using Ditto.

"""

import os
import copy
import time
import logging
import warnings
import collections

warnings.simplefilter('ignore')

import torch
from tqdm import tqdm
import numpy as np
from plato.config import Config
from plato.trainers import pers_basic
from plato.utils import optimizers
from plato.utils.arrange_saving_name import get_format_name
from plato.utils.checkpoint_operator import perform_client_checkpoint_saving
from plato.utils.checkpoint_operator import perform_client_checkpoint_loading


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def customize_train_config(self, config):
        """ Customize the training config based on the user's own requirement """

        # Note, for the ditto method, we need to set this hyper-parameter
        # to be true as the training for the local model requires the
        # initial weights of the global model before the optimization
        config['do_detailed_checkpoint'] = True
        return config

    def pers_train_one_epoch(
        self,
        config,
        kwargs,
        epoch,
        defined_model,
        pers_optimizer,
        lr_schedule,
        pers_loss_criterion,
        pers_train_loader,
        test_loader,
        epoch_loss_meter,
    ):
        """ Performing one epoch of learning for the personalization.

        In the personalization version of Ditto method, there are actually two
        types of training in the personalization stage. They are:
        A.- training with Ditto solver before final round of personalization,
            v_k = v_k - η(∇F_k(v_k) + λ(v_k - w^t))
        B.- normal training for personalization in the final round.
            v_k = v_k - η∇F_k(v_k)

        This can be witnessed in the FedRep's implementation of Ditto.
            https://github.com/lgcollins/FedRep

        Therefore, our code switches from A to B when reaching the final
            round of personalization.
        """
        personalized_model_name = config['personalized_model_name']
        model_type = config['model_name']
        current_round = kwargs['current_round']
        final_round = config['rounds']
        lamda = config['lamda']
        # As implemented in the FedRep, we do not perform the Ditto solver
        # in the final personalization round.
        # Thus, in the final round, we perform the general training for
        # the personalized model.
        if current_round < final_round:
            # 0. loaded the saved global model downloaded from the server
            #    in current round.
            #    this is actually the 'w^t' in the Algo.1 of the Ditto paper
            filename, cpk_oper = perform_client_checkpoint_loading(
                client_id=self.client_id,
                model_name=model_type,
                current_round=current_round,
                run_id=None,
                epoch=0,
                prefix=None,
                anchor_metric="round",
                mask_anchors=["personalized"],
                use_latest=False)

            # obtain the weights of the global model w^t without being
            # optimized in the training stage
            loaded_global_model_weights = cpk_oper.load_checkpoint(
                filename)['model']

            if epoch - 1 == 0:
                logging.info(
                    "[Client #%d] Loaded the unoptimized global model w^t %s for Ditto's solver",
                    self.client_id, filename)
        epoch_loss_meter.reset()
        defined_model.train()
        defined_model.to(self.device)

        pers_epochs = config["pers_epochs"]
        epoch_log_interval = pers_epochs + 1
        epoch_model_log_interval = pers_epochs + 1

        if "pers_epoch_log_interval" in config:
            epoch_log_interval = config['pers_epoch_log_interval']

        if "pers_epoch_model_log_interval" in config:
            epoch_model_log_interval = config['pers_epoch_model_log_interval']

        local_progress = tqdm(pers_train_loader,
                              desc=f'Epoch {epoch}/{pers_epochs+1}',
                              disable=True)

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)

            # backup the params of defined model before optimization
            # this is the v_k in the Algorithm. 1
            v_initial = copy.deepcopy(defined_model.state_dict())
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            ## 1.- Compute the ∇F_k(v_k), thus to compute the first term
            #   of the equation in the Algorithm. 1.
            # i.e., v_k − η∇F_k(v_k)
            # This can be achieved by the general optimization step.
            # Perfrom the training and compute the loss
            preds = defined_model(examples)
            loss = pers_loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            ## 2.- Compute the ηλ(v_k − w^t), which is the second term of
            #   the corresponding equation in Algorithm. 1.
            if current_round < final_round:
                w_net = copy.deepcopy(defined_model.state_dict())
                lr = lr_schedule.get_lr()[0]
                for key in w_net.keys():
                    w_net[key] = w_net[key] - lr * lamda * (
                        v_initial[key] - loaded_global_model_weights[key])
                defined_model.load_state_dict(w_net)

            # Update the epoch loss container
            epoch_loss_meter.update(loss.data.item(), labels.size(0))

            local_progress.set_postfix({
                'lr': lr_schedule,
                "loss": epoch_loss_meter.val,
                'loss_avg': epoch_loss_meter.avg
            })

        if (epoch - 1) % epoch_log_interval == 0 or epoch == pers_epochs:
            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id, epoch, pers_epochs, epoch_loss_meter.avg)

            output = self.perform_test_op(test_loader, defined_model)
            accuracy = output["accuracy"]

            # save the personaliation accuracy to the results dir
            self.checkpoint_personalized_accuracy(
                accuracy=accuracy,
                current_round=kwargs['current_round'],
                epoch=epoch,
                run_id=None)

        if (epoch - 1) % epoch_model_log_interval == 0 or epoch == pers_epochs:
            # the model generated during each round will be stored in the
            # checkpoints
            perform_client_checkpoint_saving(
                client_id=self.client_id,
                model_name=personalized_model_name,
                model_state_dict=self.personalized_model.state_dict(),
                config=config,
                kwargs=kwargs,
                optimizer_state_dict=pers_optimizer.state_dict(),
                lr_schedule_state_dict=lr_schedule.state_dict(),
                present_epoch=epoch,
                base_epoch=epoch,
                prefix="personalized")
