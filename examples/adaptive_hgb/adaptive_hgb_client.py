"""
A federated learning client with support for Adaptive hierarchical gradient blending.
"""

import copy
import logging
import os
import time
from dataclasses import dataclass

from plato.clients import base, simple
from plato.config import Config
from plato.models.multimodal import blending
from plato.samplers import registry as samplers_registry

#   simple.Client
#   arguments: model=None, datasource=None, algorithm=None, trainer=None
#       One can either set these four parameters in the initialization or the client will
#   define these itself based on the configuration file

#   # The functions are required by the client
#   - configure: registe the trainer and the algorithm for this client
#   - load_data: obtain the trainset and testset from the datasoruce
#   - load_payload: the algorithm will be called to get the server's model to this client
#   - train: self.trainer.train operate the local training stage


@dataclass
class Report(base.Report):
    """Report from a simple client, to be sent to the federated learning server."""
    training_time: float
    data_loading_time: float
    delta_o: float
    delta_g: float


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
        self.modality_sampler = None

        self.data_loading_time = None
        self.data_loading_time_sent = False

        model_name = Config().trainer.model_name
        filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

        save_dir = os.path.join("learningModels",
                                "client_" + str(self.client_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.recored_model_path = os.path.join(save_dir, filename)

        self.optimal_blending_weights = dict()
        # for example of the self.optimal_blending_weights:
        #   {"RGB": 0.24, "Flow": 0.48, "Audio": 0.11, "Fused"; 17}

    def record_model(self):
        """ Save the client's model to the memory """
        self.trainer.save_model(filename=self.recored_model_path)

    def load_recorded_model(self):
        """ Loaded the saved model """
        self.trainer.load_model(filename=self.recored_model_path)

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        data_loading_start_time = time.time()
        logging.info("[Client #%d] Loading its data source...", self.client_id)

        self.data_loaded = True

        logging.info("[Client #%d] Dataset size: %s", self.client_id,
                     self.datasource.num_train_examples())

        # Setting up the data sampler
        self.sampler = samplers_registry.get(self.datasource, self.client_id)

        # Setting up the modality sampler
        self.modality_sampler = samplers_registry.multimodal_get(
            datasource=self.datasource, client_id=self.client_id)

        # PyTorch uses samplers when loading data with a data loader
        self.trainset = self.datasource.get_train_set(
            self.modality_sampler.get())

        self.valset = self.datasource.get_val_set()

        if Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

        self.data_loading_time = time.time() - data_loading_start_time

    def local_global_gradient_blending(self, local_model,
                                       global_eval_avg_loses,
                                       global_eval_subtrain_avg_losses,
                                       local_eval_avg_losses,
                                       local_eval_subtrain_avg_losses):
        """ Blend the gradients for the received global model and the local model """
        # eval_avg_losses, eval_subtrainset_avg_losses,
        #   local_train_avg_losses, local_eval_avg_losses
        # obtain the global model directly as the global_model is actually the self.model
        # global_model = self.model
        # the existing modules should be the networks for modalities and the fusion net
        #   For example: ["RGB", "Flow", "Audio", "Fused"]
        existing_modules_names = global_eval_avg_loses.keys()

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
        # The main reaons is that we are afraid the client resource will be released
        #   if it is stopped
        self.load_recorded_model()
        local_model = copy.deepcopy(self.model)

        self.algorithm.load_weights(server_payload)

        # using ogr merge
        eval_avg_losses, eval_subtrainset_avg_losses, \
            local_eval_avg_losses, \
                local_train_avg_losses = self.trainer.obtain_local_global_ogr_items(
            trainset=self.trainset, evalset=self.evalset)

        self.local_global_gradient_blending(
            local_model=local_model,
            global_eval_avg_loses=eval_avg_losses,
            global_eval_subtrain_avg_losses=eval_subtrainset_avg_losses,
            local_eval_avg_losses=local_eval_avg_losses,
            local_eval_subtrain_avg_losses=local_train_avg_losses)

        self.optimal_blending_weights = self.adaptive_gradient_blending_weights(
            eval_avg_losses=eval_avg_losses,
            eval_train_avg_losses=eval_subtrainset_avg_losses,
            local_eval_avg_losses=local_eval_avg_losses,
            local_train_avg_losses=local_train_avg_losses)

    def adaptive_gradient_blending_weights(self, eval_avg_losses,
                                           eval_train_avg_losses,
                                           local_eval_avg_losses,
                                           local_train_avg_losses):
        """ Obtain the gradient blending weights """
        modalities_losses_n = {
            "eval": local_eval_avg_losses,
            "train": local_train_avg_losses
        }
        modalities_losses_N = {
            "eval": eval_avg_losses,
            "train": eval_train_avg_losses
        }
        optimal_weights = blending.get_optimal_gradient_blend_weights(
            modalities_losses_n, modalities_losses_N)

        return optimal_weights

    def obtain_delta_og(self):
        """ Compute the overfitting-generalization-ratio """
        start_eval_loss = self.trainer.global_losses_trajectory["eval"][0]
        start_train_loss = self.trainer.global_losses_trajectory["train"][0]
        end_eval_loss = self.trainer.global_losses_trajectory["eval"][-1]
        end_train_loss = self.trainer.global_losses_trajectory["train"][-1]

        delta_o = blending.compute_delta_overfitting_O(
            n_eval_avg_loss=start_eval_loss,
            n_train_avg_loss=start_train_loss,
            N_eval_avg_loss=end_eval_loss,
            N_train_avg_loss=end_train_loss)
        delta_g = blending.compute_delta_generalization(
            eval_avg_loss_n=start_eval_loss, eval_avg_loss_N=end_eval_loss)

        return delta_o, delta_g

    async def train(self):
        """The machine learning training workload on a client."""
        training_start_time = time.time()
        logging.info("[Client #%d] Started training.", self.client_id)

        # Perform model training
        if not self.trainer.train(self.trainset, self.evalset, self.sampler,
                                  self.optimal_blending_weights):
            # Training failed
            await self.sio.disconnect()

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Obtain the delta O and delta G
        delta_o, delta_g = self.obtain_delta_og()

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
                      data_loading_time, delta_o, delta_g), weights
