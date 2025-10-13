"""
A self-supervised federated learning trainer with Calibre.
"""

import logging
import os

import torch
from calibre_dataloader_strategy import CalibreDataLoaderStrategy
from calibre_loss import CalibreLoss
from calibre_lr_scheduler_strategy import CalibreLRSchedulerStrategy
from calibre_optimizer_strategy import CalibreOptimizerStrategy
from clustering import kmeans_clustering

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    LossCriterionStrategy,
    ModelUpdateStrategy,
    TrainingContext,
)


class CalibreLossStrategy(LossCriterionStrategy):
    """
    Loss strategy for Calibre that computes the Calibre loss with auxiliary losses.
    """

    def __init__(self):
        """Initialize the Calibre loss strategy."""
        self._calibre_loss = None

    def setup(self, context: TrainingContext):
        """Initialize the Calibre loss criterion."""
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

        # Get the auxiliary losses which are regularizers in the objective function
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

        self._calibre_loss = CalibreLoss(
            main_loss=loss_criterion_name,
            main_loss_params=loss_criterion_params,
            auxiliary_losses=auxiliary_losses,
            auxiliary_loss_params=auxiliary_loss_params,
            losses_weight=losses_weight,
            device=context.device,
        )

    def compute_loss(self, outputs, labels, context: TrainingContext):
        """Compute Calibre loss."""
        if isinstance(outputs, (list, tuple)):
            return self._calibre_loss(*outputs, labels=labels)
        else:
            return self._calibre_loss(outputs, labels=labels)


class CalibreDivergenceStrategy(ModelUpdateStrategy):
    """
    Model update strategy that computes and saves divergence rate after training.
    """

    def compute_divergence_rate(self, encodings, device):
        """
        Compute the divergence rate, which is the normalized distance between the points
        and the corresponding centroid.
        """
        cluster_ids_x, cluster_centers = kmeans_clustering(encodings, n_clusters=10)
        cluster_ids = torch.unique(cluster_ids_x, return_counts=False)
        cluster_divergence = torch.zeros(size=(len(cluster_ids),), device=device)
        for cluster_id in cluster_ids:
            cluster_center = cluster_centers[cluster_id]
            cluster_elems = encodings[cluster_ids_x == cluster_id]
            distance = torch.norm(cluster_elems - cluster_center, dim=1)
            divergence = torch.mean(distance)
            cluster_divergence[cluster_id] = divergence

        return torch.mean(cluster_divergence)

    def on_train_end(self, context: TrainingContext):
        """
        Compute divergence rate based on the learned features of local samples
        after training. The computed value will be saved to disk to be loaded
        when the client sends it to the server.
        """
        # Get personalized trainset from context state
        personalized_trainset = context.state.get("personalized_trainset")
        sampler = context.state.get("sampler")

        if personalized_trainset is None:
            logging.warning(
                "[Client #%d] No personalized trainset found in context.",
                context.client_id,
            )
            return

        # Handle Plato Sampler objects that have a get() method
        if sampler is not None and hasattr(sampler, "get") and callable(sampler.get):
            sampler = sampler.get()

        personalized_train_loader = torch.utils.data.DataLoader(
            dataset=personalized_trainset,
            shuffle=False,
            batch_size=10,
            sampler=sampler,
        )

        logging.info("[Client #%d] Computing the divergence rate.", context.client_id)

        sample_encodings = None

        with torch.no_grad():
            for examples, _ in personalized_train_loader:
                examples = examples.to(context.device)
                features = context.model.encoder(examples)

                sample_encodings = (
                    features
                    if sample_encodings is None
                    else torch.cat((sample_encodings, features), dim=0)
                )

        divergence_rate = self.compute_divergence_rate(sample_encodings, context.device)

        # Save the divergence
        model_path = Config().params["model_path"]
        filename = f"client_{context.client_id}_divergence_rate.pth"
        save_path = os.path.join(model_path, filename)

        torch.save(divergence_rate.detach().cpu(), save_path)


class Trainer(ComposableTrainer):
    """
    A trainer with Calibre, which computes Calibre's loss and computes the
    divergence of clusters, showing the normalized distance between the points
    and the centroid.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Calibre trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=CalibreLossStrategy(),
            optimizer_strategy=CalibreOptimizerStrategy(),
            lr_scheduler_strategy=CalibreLRSchedulerStrategy(),
            model_update_strategy=CalibreDivergenceStrategy(),
            data_loader_strategy=CalibreDataLoaderStrategy(),
        )

        # Datasets for personalization (required by SSL client)
        self.personalized_trainset = None
        self.personalized_testset = None

        # Define the personalized model (local layers)
        # This is initialized after the model is available in setup
        self.local_layers = None

    def set_personalized_datasets(self, trainset, testset):
        """
        Set the personalized trainset and testset.

        This method is called by the SSL client to provide datasets
        for the personalization phase.

        Args:
            trainset: Training dataset for personalization
            testset: Test dataset for personalization
        """
        self.personalized_trainset = trainset
        self.personalized_testset = testset

    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Test the model - uses encoder + local_layers for SSL personalization testing.

        Args:
            config: Configuration dictionary
            testset: Test dataset
            sampler: Optional sampler
            **kwargs: Additional arguments

        Returns:
            Test accuracy
        """
        # Only test during personalization phase (after SSL training rounds)
        if self.current_round <= Config().trainer.rounds:
            # During SSL training, we don't have a standard test
            # The SSL framework uses KNN or other methods separately
            self.accuracy = 0.0
            return 0.0

        # Test the personalized model (encoder + local_layers)
        if self.local_layers is None:
            logging.warning(
                "[Client #%d] No local_layers for testing.", self.client_id
            )
            self.accuracy = 0.0
            return 0.0

        batch_size = config["batch_size"]

        self.local_layers.eval()
        self.local_layers.to(self.device)

        self.model.eval()
        self.model.to(self.device)

        # Handle Plato Sampler objects
        if sampler is not None and hasattr(sampler, 'get') and callable(sampler.get):
            sampler = sampler.get()

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                # Use encoder to extract features, then classify with local_layers
                features = self.model.encoder(examples)
                outputs = self.local_layers(features)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        self.accuracy = accuracy  # Set self.accuracy for the framework
        return accuracy

    def train(self, trainset, sampler, **kwargs):
        """
        Train the model and store necessary data in context for divergence computation.

        Args:
            trainset: Training dataset
            sampler: Data sampler for this client
            **kwargs: Additional arguments including personalized_trainset

        Returns:
            Training time in seconds
        """
        # Initialize local_layers if not done yet and model has encoder
        if self.local_layers is None and hasattr(self.model, "encoder"):
            from plato.models import registry as models_registry

            model_params = Config().parameters.personalization.model._asdict()
            model_params["input_dim"] = self.model.encoder.encoding_dim
            model_params["output_dim"] = model_params["num_classes"]
            self.local_layers = models_registry.get(
                model_name=Config().algorithm.personalization.model_name,
                model_type=Config().algorithm.personalization.model_type,
                model_params=model_params,
            )

        # Store local_layers in context for optimizer strategy
        if self.local_layers is not None:
            self.context.state["local_layers"] = self.local_layers

        # Store personalized trainset and sampler in context for divergence computation
        if self.personalized_trainset is not None:
            self.context.state["personalized_trainset"] = self.personalized_trainset
            self.context.state["sampler"] = sampler

        # Call parent train method
        return super().train(trainset, sampler, **kwargs)
