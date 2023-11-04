"""
A self-supervised federated learning trainer with Calibre.
"""

import os
import logging

import torch

from plato.trainers import self_supervised_learning as ssl_trainer
from plato.config import Config

from calibre_loss import CalibreLoss
from clustering import kmeans_clustering


class Trainer(ssl_trainer.Trainer):
    """
    A trainer with Calibre, which computes Calibre's loss and computes the
    divergence of clusters, showing the normalized distance between the points
    and the centroid.
    """

    def get_ssl_criterion(self):
        """Get the loss of Calibre."""

        # Get the main loss criterion
        loss_criterion_name = (
            Config().trainer.loss_criterion
            if hasattr(Config.trainer, "loss_criterion")
            else "CrossEntropyLoss"
        )
        loss_criterion_params = (
            Config().parameters.loss_criterion._asdict()
            if hasattr(Config.parameters, "loss_criterion")
            else {}
        )

        # Get the auxiliary losses which are regularizers in the
        # objective funct
        auxiliary_losses = (
            Config().algorithm.auxiliary_loss_criterions
            if hasattr(Config.algorithm, "auxiliary_loss_criterions")
            else []
        )
        auxiliary_loss_params = (
            Config().algorithm.auxiliary_loss_criterions_param._asdict()
            if hasattr(Config.algorithm, "auxiliary_loss_criterions_param")
            else {}
        )

        # Get the weight for these losses
        losses_weight = (
            Config().algorithm.losses_weight
            if hasattr(Config.algorithm, "losses_weight")
            else {}
        )

        defined_ssl_loss = CalibreLoss(
            main_loss=loss_criterion_name,
            main_loss_params=loss_criterion_params,
            auxiliary_losses=auxiliary_losses,
            auxiliary_loss_params=auxiliary_loss_params,
            losses_weight=losses_weight,
            device=self.device,
        )

        def compute_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return defined_ssl_loss(*outputs, labels=labels)
            else:
                return defined_ssl_loss(outputs, labels=labels)

        return compute_loss

    def compute_divergence_rate(self, encodings):
        """
        Compute the divergence rate, which is the normalized distance between the points
        and the corresponding centroid.
        """
        cluster_ids_x, cluster_centers = kmeans_clustering(encodings, n_clusters=10)
        cluster_ids = torch.unique(cluster_ids_x, return_counts=False)
        cluster_divergence = torch.zeros(size=(len(cluster_ids),), device=self.device)
        for cluster_id in cluster_ids:
            cluster_center = cluster_centers[cluster_id]
            cluster_elems = encodings[cluster_ids_x == cluster_id]
            distance = torch.norm(cluster_elems - cluster_center, dim=1)
            divergence = torch.mean(distance)
            cluster_divergence[cluster_id] = divergence

        return torch.mean(cluster_divergence)

    def get_optimizer(self, model):
        """Get the optimizer"""
        optimizer = super().get_optimizer(model)
        if self.current_round > Config().trainer.rounds:
            # Add another self.model's parameters to the existing optimizer
            # This will not be trained but only used to build an entire model
            # encoder - linear classifier
            optimizer.add_param_group({"params": self.model.encoder.parameters()})
        return optimizer

    def train_run_end(self, config):
        """
        Compute divergence rate based on the learned features of local samples
        after training. The, the computed value will be saved to disk to be loaded
        when the client sends it to the server.
        """
        super().train_run_end(config)

        personalized_train_loader = torch.utils.data.DataLoader(
            dataset=self.personalized_trainset,
            shuffle=False,
            batch_size=10,
            sampler=self.sampler,
        )

        logging.info("[Client #%d] Computing the divergence rate.", self.client_id)

        sample_encodings = None

        with torch.no_grad():
            for examples, _ in personalized_train_loader:
                examples = examples.to(self.device)
                features = self.model.encoder(examples)

                sample_encodings = (
                    features
                    if sample_encodings is None
                    else torch.cat((sample_encodings, features), dim=0)
                )

        divergence_rate = self.compute_divergence_rate(sample_encodings)

        # Save the divergence
        model_path = Config().params["model_path"]
        filename = f"client_{self.client_id}_divergence_rate.pth"
        save_path = os.path.join(model_path, filename)

        torch.save(divergence_rate.detach().cpu(), save_path)
