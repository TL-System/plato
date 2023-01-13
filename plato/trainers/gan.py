"""
The training and testing loops for GAN models.

Reference:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import logging
import math
import os

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.trainers import optimizers


class Trainer(basic.Trainer):
    """A federated learning trainer for GAN models."""

    def __init__(self, model=None, **kwargs):
        super().__init__()

        if model is None:
            model = models_registry.get()
        gan_model = model
        self.generator = gan_model.generator
        self.discriminator = gan_model.discriminator
        self.loss_criterion = gan_model.loss_criterion
        self.model = gan_model

        # Use the pre-trained InceptionV3 model as a feature extractor
        # for testing
        self.inception_model = torchvision.models.inception_v3(
            pretrained=True, aux_logits=False
        )
        # Remove the last output layer of inception
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()

        self.training_start_time = 0

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            net_gen_path = f"{model_path}/Generator_{filename}"
            net_disc_path = f"{model_path}/Discriminator_{filename}"
        else:
            net_gen_path = f"{model_path}/Generator_{model_name}.pth"
            net_disc_path = f"{model_path}/Discriminator_{model_name}.pth"

        torch.save(self.generator.state_dict(), net_gen_path)
        torch.save(self.discriminator.state_dict(), net_disc_path)

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
        """Loading pre-trained model weights from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            net_gen_path = f"{model_path}/Generator_{filename}"
            net_disc_path = f"{model_path}/Discriminator_{filename}"
        else:
            net_gen_path = f"{model_path}/Generator_{model_name}.pth"
            net_disc_path = f"{model_path}/Discriminator_{model_name}.pth"

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

        self.generator.load_state_dict(torch.load(net_gen_path))
        self.discriminator.load_state_dict(torch.load(net_disc_path))

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.

        Returns:
        float: The training time.
        """
        batch_size = config["batch_size"]
        log_interval = 10

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        train_loader = torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

        self.model.to(self.device)
        self.model.train()

        # self.generator.apply(self.model.weights_init)
        # self.discriminator.apply(self.model.weights_init)

        optimizer_gen = optimizers.get(self.generator)
        optimizer_disc = optimizers.get(self.discriminator)

        real_label = 1.0
        fake_label = 0.0

        epochs = config["epochs"]
        for epoch in range(1, epochs + 1):
            # Here we assume the data samples still have labels attached to them,
            # but GAN training does not need labels, so we'll just discard them
            for batch_id, (examples, _) in enumerate(train_loader):
                cur_batch_size = len(examples)
                examples = examples.to(self.device)
                label = torch.full((cur_batch_size,), real_label, dtype=torch.float)
                label = label.to(self.device)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                optimizer_disc.zero_grad()
                # Forward pass real batch through D
                output = self.discriminator(examples).view(-1)
                # Calculate loss on all-real batch
                err_disc_real = self.loss_criterion(output, label)
                # Calculate gradients for D in backward pass
                err_disc_real.backward()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(
                    cur_batch_size, self.model.nz, 1, 1, device=self.device
                )
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                err_disc_fake = self.loss_criterion(output, label)
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
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                err_gen = self.loss_criterion(output, label)
                # Calculate gradients for G
                err_gen.backward()
                # Update G
                optimizer_gen.step()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tGenerator Loss: %.6f\t"
                            "Discriminator Loss: %.6f",
                            os.getpid(),
                            epoch,
                            epochs,
                            batch_id,
                            len(train_loader),
                            err_gen.data.item(),
                            err_disc_total.data.item(),
                        )
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tGenerator Loss: %.6f\t"
                            "Discriminator Loss: %.6f",
                            self.client_id,
                            epoch,
                            epochs,
                            batch_id,
                            len(train_loader),
                            err_gen.data.item(),
                            err_disc_total.data.item(),
                        )

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Test the Generator model with the Frechet Inception Distance metric."""

        self.model.to(self.device)
        self.model.eval()

        perplexity = -1

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config["batch_size"], shuffle=True
        )

        real_features, fake_features = [], []
        with torch.no_grad():
            for real_examples, _ in test_loader:
                real_examples = real_examples.to(self.device)

                noise = torch.randn(
                    config["batch_size"], self.model.nz, 1, 1, device=self.device
                )
                fake_examples = self.generator(noise)

                # Extract the feature of real and synthetic data with
                # InceptionV3 model pre-trained on ImageNet
                self.inception_model.to(self.device)
                feature_real = self.feature_extractor(real_examples)
                feature_fake = self.feature_extractor(fake_examples)

                # Store the feature of every real and synthetic data
                real_features.extend(list(feature_real))
                fake_features.extend(list(feature_fake))

            real_features, fake_features = np.stack(real_features), np.stack(
                fake_features
            )
            # Calculate the Frechet Distance between the feature distribution
            # of real data from testset and the feature distribution of data
            # generated by the generator.
            perplexity = self.calculate_fid(real_features, fake_features)

        return perplexity

    def feature_extractor(self, inputs):
        """Extract the feature of input data with InceptionV3.

        The feature extracted from each input is a NumPy array
        of length 2048.
        """
        # Since the input to InceptionV3 needs to be at least 75x75,
        # we will pad the input image if needed.
        hpad = math.ceil((75 - inputs.size(dim=-2)) / 2)
        vpad = math.ceil((75 - inputs.size(dim=-1)) / 2)
        hpad, vpad = max(0, hpad), max(0, vpad)
        pad = nn.ZeroPad2d((hpad, hpad, vpad, vpad))
        inputs = pad(inputs)

        # Extract feature with InceptionV3
        features = None
        with torch.no_grad():
            features = self.inception_model(inputs)
        features = features.cpu()
        features = np.array(features)

        return features

    def calculate_fid(self, real_features, fake_features):
        """Calculate the Frechet Inception Distance (FID) between the
        given real data feature and the synthetic data feature.

        A lower FID indicates a better Generator model.

        The implementation is borrowed from the following link:
        https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
        """
        # calculate mean and covariance statistics
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
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
