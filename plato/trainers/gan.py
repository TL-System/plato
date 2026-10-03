"""
The training and testing loops for GAN models.

Reference:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import logging
import math
import os
import pickle
from collections.abc import Callable
from typing import Optional, cast

import numpy as np
import scipy
import torch
import torch.nn as nn
import torchvision

from plato.callbacks.trainer import TrainerCallback, resolve_num_samples
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import optimizers
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    OptimizerStrategy,
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)
from plato.utils.checkpoint_paths import (
    checkpoint_name,
    checkpoint_path,
    checkpoint_sidecar,
)


class GANOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy for GAN training with separate optimizers
    for generator and discriminator.

    This strategy creates and manages two optimizers instead of one,
    which is required for GAN training where generator and discriminator
    are trained alternately.
    """

    def __init__(self):
        """Initialize GAN optimizer strategy."""
        self.optimizer_gen = None
        self.optimizer_disc = None

    def create_optimizer(
        self, model: nn.Module, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """
        Create optimizers for both generator and discriminator.

        Note: Returns the discriminator optimizer as the "primary" optimizer,
        but both are stored and used by GANTrainingStepStrategy.
        """
        # Create separate optimizers for generator and discriminator
        self.optimizer_gen = optimizers.get(model.generator)
        self.optimizer_disc = optimizers.get(model.discriminator)

        # Store both optimizers in context for use by training step strategy
        context.state["optimizer_gen"] = self.optimizer_gen
        context.state["optimizer_disc"] = self.optimizer_disc

        # Return discriminator optimizer as the "primary" one
        return self.optimizer_disc


class GANTrainingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy for GAN models.

    This implements the standard GAN training procedure:
    1. Update discriminator with real data
    2. Update discriminator with fake data
    3. Update generator to fool discriminator

    The strategy retrieves the dual optimizers from the context and
    performs the alternating updates.
    """

    def training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        examples: torch.Tensor,
        labels: torch.Tensor,
        loss_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        context: TrainingContext,
    ) -> torch.Tensor:
        """
        Perform one GAN training step.

        Args:
            model: GAN model with generator and discriminator
            optimizer: Not used (we use the ones from context)
            examples: Real data batch
            labels: Not used (GAN doesn't need labels)
            loss_criterion: GAN loss criterion
            context: Training context with stored optimizers

        Returns:
            Combined loss for logging (discriminator + generator)
        """
        # Retrieve optimizers from context
        optimizer_gen = context.state["optimizer_gen"]
        optimizer_disc = context.state["optimizer_disc"]

        real_label = 1.0
        fake_label = 0.0

        cur_batch_size = len(examples)
        device = context.device

        # Create label tensors
        label = torch.full((cur_batch_size,), real_label, dtype=torch.float)
        label = label.to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        optimizer_disc.zero_grad()

        # Forward pass real batch through D
        discriminator = getattr(model, "discriminator", None)
        generator = getattr(model, "generator", None)
        loss_fn = getattr(model, "loss_criterion", loss_criterion)
        latent_dim = getattr(model, "nz", None)

        if not callable(discriminator) or not callable(generator):
            raise TypeError(
                "GAN models must define callable 'discriminator' and 'generator'."
            )
        if not callable(loss_fn):
            raise TypeError("GAN model must provide a callable 'loss_criterion'.")
        if latent_dim is None:
            raise AttributeError("GAN model must expose 'nz' latent dimension.")

        discriminator_fn = cast(Callable[[torch.Tensor], torch.Tensor], discriminator)
        generator_fn = cast(Callable[[torch.Tensor], torch.Tensor], generator)
        loss_callable = cast(
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor], loss_fn
        )

        output = discriminator_fn(examples).view(-1)

        # Calculate loss on all-real batch
        err_disc_real = loss_callable(output, label)

        # Calculate gradients for D in backward pass
        err_disc_real.backward()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(cur_batch_size, int(latent_dim), 1, 1, device=device)

        # Generate fake image batch with G
        fake = generator_fn(noise)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = discriminator_fn(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        err_disc_fake = loss_callable(output, label)

        # Calculate the gradients for this batch, accumulated (summed)
        # with previous gradients
        err_disc_fake.backward()

        # Compute error of D as sum over the fake and the real batches
        err_disc_total = err_disc_real + err_disc_fake

        # Update D
        optimizer_disc.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator_fn(fake).view(-1)

        # Calculate G's loss based on this output
        err_gen = loss_callable(output, label)

        # Calculate gradients for G
        err_gen.backward()

        # Update G
        optimizer_gen.step()

        # Store individual losses in context for logging
        context.state["last_gen_loss"] = err_gen.item()
        context.state["last_disc_loss"] = err_disc_total.item()

        # Return combined loss for tracking
        return err_disc_total + err_gen


class GANLoggingCallback(TrainerCallback):
    """
    Callback for logging GAN-specific training progress.

    Logs both generator and discriminator losses during training.
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Log dataset loading at start of training."""
        num_samples = resolve_num_samples(trainer)
        if num_samples is not None:
            message = f"Loading the dataset with size {num_samples}."
        else:
            message = "Loading the dataset."

        if trainer.client_id == 0:
            logging.info(
                "[Server #%s] %s",
                os.getpid(),
                message,
            )
        else:
            logging.info(
                "[Client #%d] %s",
                trainer.client_id,
                message,
            )

    def on_train_step_end(self, trainer, config, batch=None, loss=None, **kwargs):
        """Log generator and discriminator losses."""
        log_interval = 10

        if isinstance(batch, int) and log_interval > 0 and batch % log_interval == 0:
            # Retrieve GAN-specific losses from context
            gen_loss = trainer.context.state.get("last_gen_loss", 0.0)
            disc_loss = trainer.context.state.get("last_disc_loss", 0.0)

            if trainer.client_id == 0:
                logging.info(
                    "[Server #%d] Epoch: [%d/%d][%d/%d]\tGenerator Loss: %.6f\t"
                    "Discriminator Loss: %.6f",
                    os.getpid(),
                    trainer.current_epoch,
                    config["epochs"],
                    batch,
                    len(trainer.train_loader),
                    gen_loss,
                    disc_loss,
                )
            else:
                logging.info(
                    "[Client #%d] Epoch: [%d/%d][%d/%d]\tGenerator Loss: %.6f\t"
                    "Discriminator Loss: %.6f",
                    trainer.client_id,
                    trainer.current_epoch,
                    config["epochs"],
                    batch,
                    len(trainer.train_loader),
                    gen_loss,
                    disc_loss,
                )


class GANTestingStrategy(TestingStrategy):
    """
    Testing strategy for GAN models using Frechet Inception Distance (FID).

    This strategy evaluates GAN generators by:
    1. Generating fake images from random noise
    2. Extracting features from both real and fake images using InceptionV3
    3. Computing FID metric between feature distributions

    A lower FID indicates better generator quality.
    """

    def __init__(self):
        """Initialize GAN testing strategy with InceptionV3 feature extractor."""
        # Use the pre-trained InceptionV3 model as a feature extractor for testing
        self.inception_model = torchvision.models.inception_v3(
            weights=torchvision.models.Inception_V3_Weights.DEFAULT
        )
        # Remove the last output layer of inception and auxiliary classifier
        self.inception_model.fc = nn.Identity()
        self.inception_model.AuxLogits = None
        self.inception_model.eval()

    def test_model(self, model, config, testset, sampler, context):
        """
        Test the Generator model with the Frechet Inception Distance metric.

        Args:
            model: GAN model with generator
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional evaluation partition sampler
            context: Training context with device info

        Returns:
            FID score (float) - lower is better
        """
        model.to(context.device)
        model.eval()

        perplexity = -1

        sampler_obj = sampler
        if sampler is not None and not isinstance(sampler, torch.utils.data.Sampler):
            get_sampler = getattr(sampler, "get", None)
            if callable(get_sampler):
                sampler_obj = get_sampler()
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config["batch_size"], shuffle=False, sampler=sampler_obj
        )

        real_features, fake_features = [], []
        with torch.no_grad():
            for real_examples, _ in test_loader:
                real_examples = real_examples.to(context.device)

                noise = torch.randn(
                    len(real_examples), model.nz, 1, 1, device=context.device
                )
                fake_examples = model.generator(noise)

                # Extract the feature of real and synthetic data with
                # InceptionV3 model pre-trained on ImageNet
                self.inception_model.to(context.device)
                feature_real = self._feature_extractor(real_examples, context.device)
                feature_fake = self._feature_extractor(fake_examples, context.device)

                # Store the feature of every real and synthetic data
                real_features.extend(list(feature_real))
                fake_features.extend(list(feature_fake))

            if len(real_features) < 2:
                raise ValueError("GAN FID evaluation requires at least two samples.")
            real_features, fake_features = (
                np.stack(real_features),
                np.stack(fake_features),
            )
            # Calculate the Frechet Distance between the feature distribution
            # of real data from testset and the feature distribution of data
            # generated by the generator.
            perplexity = self._calculate_fid(real_features, fake_features)

        return perplexity

    def _feature_extractor(self, inputs, device):
        """
        Extract features from input data using InceptionV3.

        The feature extracted from each input is a NumPy array
        of length 2048.

        Args:
            inputs: Input images tensor
            device: Device to run feature extraction on

        Returns:
            NumPy array of features
        """
        # Since the input to InceptionV3 needs to be at least 75x75,
        # we will pad the input image if needed.
        hpad = math.ceil((75 - inputs.size(dim=-1)) / 2)
        vpad = math.ceil((75 - inputs.size(dim=-2)) / 2)
        hpad, vpad = max(0, hpad), max(0, vpad)
        pad = nn.ZeroPad2d((hpad, hpad, vpad, vpad))
        inputs = pad(inputs)

        # Extract feature with InceptionV3
        features = None
        with torch.no_grad():
            features = self.inception_model(inputs)
        return features.detach().cpu().numpy()

    def _calculate_fid(self, real_features, fake_features):
        """
        Calculate the Frechet Inception Distance (FID) between the
        given real data feature and the synthetic data feature.

        A lower FID indicates a better Generator model.

        The implementation is borrowed from the following link:
        https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI

        Args:
            real_features: Features extracted from real images
            fake_features: Features extracted from generated images

        Returns:
            FID score (float)
        """
        # calculate mean and covariance statistics
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid


class Trainer(ComposableTrainer):
    """A federated learning trainer for GAN models using composable architecture."""

    def __init__(self, model=None, callbacks=None):
        """
        Initialize GAN trainer with GAN-specific strategies.

        Args:
            model: GAN model (or None to use registry)
            callbacks: Additional callbacks beyond GAN-specific ones
        """
        # Prepare GAN-specific callbacks
        gan_callbacks = [GANLoggingCallback]
        if callbacks is not None:
            gan_callbacks.extend(callbacks)

        # Initialize with GAN-specific strategies
        super().__init__(
            model=model,
            callbacks=gan_callbacks,
            optimizer_strategy=GANOptimizerStrategy(),
            training_step_strategy=GANTrainingStepStrategy(),
            testing_strategy=GANTestingStrategy(),
            loss_strategy=None,  # Not used for GAN
            lr_scheduler_strategy=None,  # Use default
            model_update_strategy=None,  # Use default
            data_loader_strategy=None,  # Use default
        )

        # GAN-specific attributes
        model_instance = self._require_model()
        self.generator = getattr(model_instance, "generator")
        self.discriminator = getattr(model_instance, "discriminator")

    def save_model(self, filename=None, location=None):
        """
        Save the GAN model (both generator and discriminator) to files.

        Args:
            filename: Optional filename (without path)
            location: Optional directory path
        """
        # Worker and urgent snapshots share the ordinary tree reader contract.
        # Historical default/explicit .pth pairs retain their existing format.
        if filename is not None and filename.endswith(".safetensors"):
            return super().save_model(filename, location)
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            # Resolve the explicit anchor before adding companion prefixes.
            # Otherwise 'nested/../pair.pth' aliases both networks to one file,
            # and an escaping anchor fails only after writing the pair.
            filename = os.path.relpath(
                checkpoint_path(model_path, filename), os.path.realpath(model_path)
            )
            net_gen_path = checkpoint_path(model_path, f"Generator_{filename}")
            net_disc_path = checkpoint_path(model_path, f"Discriminator_{filename}")
        else:
            net_gen_path = checkpoint_path(
                model_path, checkpoint_name("Generator", model_name, suffix=".pth")
            )
            net_disc_path = checkpoint_path(
                model_path, checkpoint_name("Discriminator", model_name, suffix=".pth")
            )

        history_name = filename if filename is not None else checkpoint_name(
            model_name, suffix=".pth"
        )
        history_path = checkpoint_sidecar(
            model_path, checkpoint_path(model_path, history_name)
        )
        os.makedirs(os.path.dirname(net_gen_path), exist_ok=True)
        os.makedirs(os.path.dirname(net_disc_path), exist_ok=True)
        torch.save(self.generator.state_dict(), net_gen_path)
        torch.save(self.discriminator.state_dict(), net_disc_path)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "wb") as history_file:
            pickle.dump(self.run_history, history_file)

        if self.client_id == 0:
            logging.info(
                "[Server #%d] Generator Model saved to %s.", os.getpid(), net_gen_path
            )
            logging.info(
                "[Server #%d] Discriminator Model saved to %s.",
                os.getpid(),
                net_disc_path,
            )
        else:
            logging.info(
                "[Client #%d] Generator Model saved to %s.",
                self.client_id,
                net_gen_path,
            )
            logging.info(
                "[Client #%d] Discriminator Model saved to %s.",
                self.client_id,
                net_disc_path,
            )

    def load_model(self, filename=None, location=None):
        """
        Load pre-trained GAN model weights from files.

        Args:
            filename: Optional filename (without path)
            location: Optional directory path
        """
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name
        if (filename is not None and filename.endswith(".safetensors")
                and os.path.isfile(checkpoint_path(model_path, filename))):
            return super().load_model(filename, location)

        if filename is not None:
            filename = os.path.relpath(
                checkpoint_path(model_path, filename), os.path.realpath(model_path)
            )
            net_gen_path = checkpoint_path(model_path, f"Generator_{filename}")
            net_disc_path = checkpoint_path(model_path, f"Discriminator_{filename}")
        else:
            net_gen_path = checkpoint_path(
                model_path, checkpoint_name("Generator", model_name, suffix=".pth")
            )
            net_disc_path = checkpoint_path(
                model_path, checkpoint_name("Discriminator", model_name, suffix=".pth")
            )

        if self.client_id == 0:
            logging.info(
                "[Server #%d] Loading a Generator model from %s.",
                os.getpid(),
                net_gen_path,
            )
            logging.info(
                "[Server #%d] Loading a Discriminator model from %s.",
                os.getpid(),
                net_disc_path,
            )
        else:
            logging.info(
                "[Client #%d] Loading a Generator model from %s.",
                self.client_id,
                net_gen_path,
            )
            logging.info(
                "[Client #%d] Loading a Discriminator model from %s.",
                self.client_id,
                net_disc_path,
            )

        history_name = filename if filename is not None else checkpoint_name(
            model_name, suffix=".pth"
        )
        history_path = checkpoint_sidecar(
            model_path, checkpoint_path(model_path, history_name)
        )
        # GAN checkpoints historically contain torch.save data, including
        # worker files whose supplied name ends in .safetensors. File handles
        # avoid torch.load's suffix-based Safetensors dispatch for these files.
        with open(net_gen_path, "rb") as generator_file:
            generator_state = torch.load(
                generator_file, map_location="cpu", weights_only=True
            )
        with open(net_disc_path, "rb") as discriminator_file:
            discriminator_state = torch.load(
                discriminator_file, map_location="cpu", weights_only=True
            )
        self.generator.load_state_dict(generator_state)
        self.discriminator.load_state_dict(discriminator_state)
        if os.path.isfile(history_path):
            with open(history_path, "rb") as history_file:
                self.run_history = pickle.load(history_file)

    def pause_training(self):
        """Remove this worker's ordinary artifacts and any historical GAN pair."""
        super().pause_training()
        if hasattr(Config().trainer, "max_concurrency"):
            worker = checkpoint_name(Config().trainer.model_name, self.client_id,
                                     Config().params["run_id"], suffix=".safetensors")
            for prefix in ("Generator", "Discriminator"):
                path = checkpoint_path(Config().params["model_path"], f"{prefix}_{worker}")
                if os.path.isfile(path):
                    os.remove(path)
