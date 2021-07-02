#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A federated learning client with support for Adaptive gradient blending.

"""
import os

import copy
import torch

from plato.config import Config

from plato.clients import simple

from plato.models.multimodal import blending

#   simple.Client
#   arguments: model=None, datasource=None, algorithm=None, trainer=None
#       One can either set these four parameters in the initialization or the client will
#   define these itself based on the configuration file

#   # The functions are required by the client
#   - configure: registe the trainer and the algorithm for this client
#   - load_data: obtain the trainset and testset from the datasoruce
#   - load_payload: the algorithm will be called to get the server's model to this client
#   - train: self.trainer.train operate the local training stage


class Client(simple.Client):
    """A federated learning client with support for Adaptive gradient blending.
    """
    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__()
        self.model = model
        self.datasource = datasource
        self.algorithm = algorithm
        self.trainer = trainer
        self.trainset = None  # Training dataset
        self.valset = None  # Validation dataset
        self.testset = None  # Testing dataset
        self.sampler = None

        self.data_loading_time = None
        self.data_loading_time_sent = False

        cur_work_path = os.getcwd()
        save_path = os.path.join(cur_work_path, "learningModels",
                                 "client_" + str(self.client_id),
                                 "recored_model.pt")
        self.recored_model_path = save_path

    def record_model(self):
        # put the client's model to the desktop
        torch.save(self.model.state_dict(), self.recored_model_path)

    def load_recorded_model(self):
        self.model.load_state_dict(torch.load(self.recored_model_path))

    def local_global_gradient_blending(self, local_model,
                                       global_eval_avg_loses,
                                       global_eval_subtrain_avg_losses,
                                       local_eval_avg_losses,
                                       local_eval_subtrain_avg_losses):

        # eval_avg_losses, eval_subtrainset_avg_losses, local_train_avg_losses, local_eval_avg_losses

        # obtain the global model directly as the global_model is actually the self.model
        # global_model = self.model

        # the existing modules should be the networks for modalities and the fusion net
        #   For example: ["RGB", "Flow", "Audio", "Fused"]
        existing_modules_names = global_eval_avg_loses.keys()

        modules_merge_ratios = dict()
        for module_nm in existing_modules_names:
            md_global_eval_loss = global_eval_avg_loses[module_nm]
            md_global_eval_trainset_loss = global_eval_subtrain_avg_losses[
                module_nm]
            md_local_eval_loss = local_eval_avg_losses[module_nm]
            md_local_eval_trainset_loss = local_eval_subtrain_avg_losses[
                module_nm]
            local_global_ogr = blending.OGR_n2N(
                n_eval_avg_loss=md_local_eval_loss,
                n_train_avg_loss=md_local_eval_trainset_loss,
                N_eval_avg_loss=md_global_eval_loss,
                N_train_avg_loss=md_global_eval_trainset_loss)

            # merge the corresponding local module, global module
            global_module_wt = self.model.module_name_net_mapper[
                module_nm].weight.data
            local_module_wt = local_model.module_name_net_mapper[
                module_nm].weight.data
            merged_module_wt = local_global_ogr * local_module_wt + (
                1 - local_global_ogr) * global_module_wt

            # the reason why we set the global directly is because
            #   we want to save the space
            self.model.assing_weights(module_name=module_nm,
                                      weights=merged_module_wt)

    def load_payload(self, server_payload) -> None:
        """Loading the server model onto this client."""

        # In general, we only need to get the previous local model by:
        #   local_model = self.model
        # because the weights from the server have not been assigned to the self.model
        # But we choose a more complex way that loads the weights from the file
        # The main reaons is that we are afraid the client resource will be released if it is stopped
        self.load_recorded_model()
        local_model = copy.deepcopy(self.model)

        self.algorithm.load_weights(server_payload)

        # using ogr merge
        eval_avg_losses, eval_subtrainset_avg_losses, \
            local_train_avg_losses, local_eval_avg_losses = self.trainer.obtain_local_global_OGR_items(
            trainset=self.trainset, evalset=self.evalset)

        self.local_global_gradient_blending(
            local_model=local_model,
            global_eval_avg_loses=eval_avg_losses,
            global_eval_subtrain_avg_losses=eval_subtrainset_avg_losses,
            local_eval_avg_losses=local_train_avg_losses,
            local_eval_subtrain_avg_losses=local_eval_avg_losses)

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()
        logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        if not self.trainer.train(self.trainset, self.evalset, self.sampler):
            # Training failed
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if Config().clients.do_test:
            accuracy = self.trainer.test(self.testset)

            if accuracy == 0:
                # The testing process failed, disconnect from the server
                await self.sio.disconnect()

            logging.info("[Client #{:d}] Test accuracy: {:.2f}%".format(
                self.client_id, 100 * accuracy))
        else:
            accuracy = 0

        training_time = time.time() - training_start_time
        data_loading_time = 0

        if not self.data_loading_time_sent:
            data_loading_time = self.data_loading_time
            self.data_loading_time_sent = True

        return Report(self.sampler.trainset_size(), accuracy, training_time,
                      data_loading_time), weights
