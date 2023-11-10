"""
A dishonest split learning trainer which will try to reconstruct private data
    during split learning.
"""
import os
import logging

import torch
from torchmetrics.text.rouge import ROUGEScore

from split_learning_trainer import Trainer as HonestTrainer
from plato.config import Config
from plato.trainers import tracking


class DishonestTrainer(HonestTrainer):
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

    def train_attack_model(
        self, attack_parameters, reconstructed_data, intermediate_features
    ):
        """
        Train models for attack for one step.
        """
        if attack_parameters.method == "input_ids":
            outputs = self.model.guessed_client_model(input_ids=reconstructed_data)
        else:
            outputs = self.model.guessed_client_model(inputs_embeds=reconstructed_data)
        loss = torch.nn.functional.mse_loss(
            outputs.last_hidden_state, intermediate_features
        )
        loss.backward(retain_graph=True)
        return loss

    def attack(self, intermediate_features, labels):
        """
        In the attack for LLMs, the server have two policies to reconstruct the private data
            1. Directly reconstruct the token ids.
            2. Reconstruct the embeddings and then reconstruct the token ids.
        """
        self.model.calibrate_guessed_client()
        attack_parameters = Config().parameters.attack
        assert attack_parameters.method in ["input_ids,embeddings"]
        if attack_parameters.method == "input_ids":
            reconstructed_data = torch.zeros(
                (intermediate_features.shape[0], intermediate_features.shape[1])
            ).requires_grad_(True)
        else:
            reconstructed_data = torch.zeros(
                intermediate_features.shape
            ).requires_grad_(True)

        # Assign optimizer for attackings
        optimizer_guessed_client = torch.optim.SGD(
            self.model.guessed_client_model,
            lr=attack_parameters.optimizer.lr_guessed_client,
        )
        optimizer_reconstructed_data = torch.optim.Adam(
            [reconstructed_data],
            lr=attack_parameters.optimizer.lr_reconstructed_data,
            amsgrad=True,
        )
        scheduler_reconstructed_data = torch.optim.lr_scheduler.StepLR(
            optimizer_reconstructed_data,
            step_size=attack_parameters.scheduler.step,
            gamma=attack_parameters.scheduler.gamma,
        )
        reconstructed_data = reconstructed_data.to(self.device)
        self.model.guessed_client_model = self.model.guessed_client_model.to(
            self.device
        )
        # Server can set is as evaluation mode for ease of attack.
        self.model.guessed_client_model.eval()
        intermediate_features = intermediate_features.to(self.device)

        self._loss_tracker_reconstructed_data.reset()
        self._loss_tracker_guessed_client.reset()
        # begin Unsplit gradient descent.
        for iteration in range(attack_parameters.outer_iterations):
            # gradient descent on the reconstructed data
            for _ in range(attack_parameters.inner_iterations):
                optimizer_reconstructed_data.zero_grad()
                loss = self.train_attack_model(
                    attack_parameters, reconstructed_data, intermediate_features
                )
                self._loss_tracker_reconstructed_data.update(
                    loss, intermediate_features.size(0)
                )
                optimizer_reconstructed_data.step()
            scheduler_reconstructed_data.step()
            # gradient descent on the guessed client model
            for _ in range(attack_parameters.inner_iterations):
                optimizer_guessed_client.zero_grad()
                loss = self.train_attack_model(
                    attack_parameters, reconstructed_data, intermediate_features
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
                    self._loss_tracker_guessed_client.average(),
                )

        self.model.guessed_client_model = self.model.guessed_client_model.to(
            torch.device("cpu")
        )
        intermediate_features = intermediate_features.detach().cpu()
        # if attack method is embedding, we will generate the reconstructed input ids
        #   from the reconstructed embeddings.
        if attack_parameters == "embedding":
            embedding_weights = self.model.guessed_client.wte.weight.data
            embedding_weights = embedding_weights.reshape(
                1, 1, embedding_weights.size(0), embedding_weights.size(1)
            ).expand(
                reconstructed_data.size(0),
                reconstructed_data.size(1),
                embedding_weights.size(0),
                embedding_weights.size(1),
            )
            reconstructed_data = reconstructed_data.reshape(
                reconstructed_data.size(0),
                reconstructed_data.size(1),
                1,
                reconstructed_data.size(2),
            )
            distance = torch.norm(
                reconstructed_data - self.model.guessed_client.wte.weight.data,
                dim=3,
                p=2,
            )
            reconstructed_data = distance.argmin(dim=2)

        reconstructed_data = reconstructed_data.long()
        labels = labels.to(self.device)
        labels = labels.long()

        # calculate the evaluation metrics in attack
        evaluation_metrics = dict()
        # calculate accuracy
        accuracy = torch.sum(
            labels == reconstructed_data, dim=1
        ) / reconstructed_data.size(1)
        self.accuracy_sum += torch.sum(accuracy).item()
        self.sample_counts += reconstructed_data.size(0)
        evaluation_metrics["attack_accuracy"] = self.accuracy_sum / self.sample_counts

        predicted_text = self.tokenizer.decode(
            reconstructed_data.detach().cpu().numpy().tolist()
        )
        ground_truth = self.tokenizer.decode(labels.detach().cpu().numpy().tolist())
        self.rouge_score.update(predicted_text, ground_truth)
        evaluation_metrics["ROUGE"] = self.rouge_score.compute()
