"""
Implementation of the trainer for Calibre algorithm.

"""

import os
import logging

import torch

from plato.config import Config
from pflbases import ssl_trainer
from pflbases.filename_formatter import NameFormatter


from calibre_loss import CalibreLoss
from clustering import kmeans_clustering


class Trainer(ssl_trainer.Trainer):
    """A trainer for the Calibre method."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        self.samples_encoding = None
        self.samples_label = None

        self.clusters_divergence = None
        self.clusters_center = None
        self.divergence_rate = None

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""

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

        auxiliary_losses = (
            Config().algorithm.auxiliary_loss_criterions
            if hasattr(Config.algorithm, "auxiliary_loss_criterions")
            else []
        )
        auxiliary_losses_params = (
            Config().algorithm.auxiliary_loss_criterions_param._asdict()
            if hasattr(Config.algorithm, "auxiliary_loss_criterions_param")
            else {}
        )

        losses_weight = (
            Config().algorithm.losses_weight
            if hasattr(Config.algorithm, "losses_weight")
            else {}
        )

        defined_ssl_loss = CalibreLoss(
            main_loss=loss_criterion_name,
            main_loss_params=loss_criterion_params,
            auxiliary_losses=auxiliary_losses,
            auxiliary_losses_params=auxiliary_losses_params,
            losses_weight=losses_weight,
            device=self.device,
        )

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return defined_ssl_loss(*outputs, labels=labels)
            else:
                return defined_ssl_loss(outputs, labels=labels)

        return compute_plato_loss

    def compute_divergence_rate(self, config):
        """Computing the divergence rate of the local model"""
        cluster_ids_x, self.clusters_center = kmeans_clustering(
            self.samples_encoding, n_clusters=10
        )
        clusters_id = torch.unique(cluster_ids_x, return_counts=False)
        self.clusters_divergence = torch.zeros(
            size=(len(clusters_id),), device=self.device
        )
        for cluster_id in clusters_id:
            cluster_center = self.clusters_center[cluster_id]
            cluster_elems = self.samples_encoding[cluster_ids_x == cluster_id]
            distance = torch.norm(cluster_elems - cluster_center, dim=1)
            divergence = torch.mean(distance)
            self.clusters_divergence[cluster_id] = divergence

        self.divergence_rate = torch.mean(self.clusters_divergence)

    def get_divergence_filepath(self):
        """Get the file path of the divergence rate."""
        save_dir = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name=self.model_name,
            round_n=self.current_round,
            epoch_n=None,
            run_id=None,
            prefix="divergence",
            ext="pth",
        )

        return os.path.join(save_dir, filename)

    def save_divergences(self):
        """Saving the local divergence of the client."""

        save_path = self.get_divergence_filepath()
        logging.info(
            "[Client #%d] Saving the divergence rate to %s.", self.client_id, save_path
        )
        torch.save(self.divergence_rate.detach().cpu(), save_path)
        self.save_model

    def train_run_end(self, config):
        """Get the features of local samples after training."""
        super().train_run_end(config)

        personalized_train_loader = self.get_personalized_train_loader()

        logging.info("[Client #%d] Computing the divergence rate.", self.client_id)

        with torch.no_grad():
            for _, (examples, labels) in enumerate(personalized_train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                features = self.model.encoder(examples)

                self.samples_encoding = (
                    features
                    if self.samples_encoding is None
                    else torch.cat((self.samples_encoding, features), dim=0)
                )
                self.samples_label = (
                    labels
                    if self.samples_label is None
                    else torch.cat((self.samples_label, labels), dim=0)
                )

        self.compute_divergence_rate(config)
        self.save_divergences()
