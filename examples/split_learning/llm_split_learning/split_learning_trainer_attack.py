"""
A curious split learning trainer which will try to reconstruct private data
    during split learning.
"""

import logging
import os

import torch
from split_learning_llm_model import get_module
from split_learning_trainer import Trainer as HonestTrainer
from torchmetrics.text.rouge import ROUGEScore

from plato.config import Config
from plato.trainers import tracking


class CuriousTrainer(HonestTrainer):
    """
    The trainer will use attack function to reconstruct the private data
        based on intermedaite feature.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self._loss_tracker_guessed_client = tracking.LossTracker()
        self._loss_tracker_reconstructed_data = tracking.LossTracker()
        self.sample_counts = 0.0
        self.accuracy_sum = 0.0
        self.rouge_score = ROUGEScore()

    def train_attack_model(self, reconstructed_data, intermediate_features):
        """
        Train models for attack for one step.
        """
        outputs = self.model.guessed_client_model(inputs_embeds=reconstructed_data)
        loss = torch.nn.functional.mse_loss(outputs.logits, intermediate_features)
        loss.backward()
        return loss

    # pylint:disable=too-many-locals
    def attack(self, intermediate_features, labels):
        """
        In the attack for LLMs, the server have two policies to reconstruct the private data
            1. Directly reconstruct the token ids.
            2. Reconstruct the embeddings and then reconstruct the token ids.
        """

        attack_parameters = Config().parameters.attack
        if not (
            hasattr(attack_parameters, "calibrate_guessed_client")
            and not attack_parameters.calibrate_guessed_client
        ):
            self.model.calibrate_guessed_client()
        reconstructed_data = torch.zeros(intermediate_features.shape).requires_grad_(
            True
        )

        # Assign optimizer for attackings
        # AdamW is the optimizer of language model. In the origin Unsplit,
        #   they use SGD for ResNet.
        optimizer_guessed_client = torch.optim.AdamW(
            self.model.guessed_client_model.parameters(),
            lr=attack_parameters.optimizer.lr_guessed_client,
        )
        optimizer_reconstructed_data = torch.optim.Adam(
            [reconstructed_data],
            lr=attack_parameters.optimizer.lr_reconstructed_data,
            amsgrad=True,
        )
        reconstructed_data = reconstructed_data.to(self.device)
        self.model.guessed_client_model = self.model.guessed_client_model.to(
            self.device
        )
        # Server can set is as evaluation mode for ease of attack
        #   due to the randomness in norm layers, drop out layer.
        self.model.guessed_client_model.eval()
        intermediate_features = intermediate_features.to(self.device)

        self._loss_tracker_reconstructed_data.reset()
        self._loss_tracker_guessed_client.reset()
        # begin Unsplit gradient descent
        for iteration in range(attack_parameters.outer_iterations):
            # gradient descent on the reconstructed data
            for _ in range(attack_parameters.inner_iterations):
                optimizer_reconstructed_data.zero_grad()
                loss = self.train_attack_model(
                    reconstructed_data, intermediate_features
                )
                self._loss_tracker_reconstructed_data.update(
                    loss, intermediate_features.size(0)
                )
                optimizer_reconstructed_data.step()
            # gradient descent on the guessed client model
            for _ in range(attack_parameters.inner_iterations):
                optimizer_guessed_client.zero_grad()
                loss = self.train_attack_model(
                    reconstructed_data, intermediate_features
                )
                self._loss_tracker_guessed_client.update(
                    loss, intermediate_features.size(0)
                )
                optimizer_guessed_client.step()

            if iteration % attack_parameters.report_interval == 0:
                logging.info(
                    "[Server #%d] At iteration [%s/%s] Reconstructed data loss: %.2f. Guessed : %.2f.",
                    os.getpid(),
                    str(iteration),
                    str(attack_parameters.outer_iterations),
                    self._loss_tracker_reconstructed_data.average,
                    self._loss_tracker_guessed_client.average,
                )

        self.model.guessed_client_model = self.model.guessed_client_model.to(
            torch.device("cpu")
        )
        intermediate_features = intermediate_features.detach().cpu()
        reconstructed_data = reconstructed_data.detach().cpu()
        # We will generate the reconstructed input ids
        #   from the reconstructed embeddings.
        embedding_layer = get_module(
            self.model.guessed_client_model,
            attack_parameters.embedding_layer.split("."),
        )
        embedding_weights = embedding_layer.weight.data
        reconstructed_inputs = torch.zeros(
            reconstructed_data.size(0), reconstructed_data.size(1)
        )
        # Use for loops will decrease the speed but save the memory.
        for batch_id in range(reconstructed_data.size(0)):
            for word_id in range(reconstructed_data.size(1)):
                distance = (
                    reconstructed_data[batch_id][word_id].reshape(1, -1)
                    - embedding_weights
                )
                distance = torch.norm(
                    distance,
                    dim=1,
                    p=2,
                )
                reconstructed_inputs[batch_id][word_id] = distance.argmin(dim=0).item()

        reconstructed_inputs = reconstructed_inputs.long()
        labels = labels.to(torch.device("cpu"))
        labels = labels.long()

        # calculate the evaluation metrics in attack
        evaluation_metrics = {}
        # calculate accuracy
        accuracy = torch.sum(
            labels == reconstructed_inputs, dim=1
        ) / reconstructed_inputs.size(1)
        self.accuracy_sum += torch.sum(accuracy).item()
        self.sample_counts += reconstructed_inputs.size(0)
        evaluation_metrics["attack_accuracy"] = self.accuracy_sum / self.sample_counts
        # calculate Rouge scores
        predicted_text = self.tokenizer.batch_decode(reconstructed_inputs)
        ground_truth = self.tokenizer.batch_decode(labels.detach().cpu())
        self.rouge_score.update(predicted_text, ground_truth)
        evaluation_metrics["ROUGE"] = self.rouge_score.compute()
        return evaluation_metrics
