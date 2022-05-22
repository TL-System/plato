"""
The training and testing loops for GAN models.

Reference:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import logging
import os

import torch

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A federated learning trainer for GAN models."""

    def __init__(self, model=None):
        super().__init__()

        if model is None:
            model = models_registry.get()
        gan_model = model
        self.generator = gan_model.generator
        self.discriminator = gan_model.discriminator
        self.loss_criterion = gan_model.loss_criterion
        self.model = gan_model

        self.training_start_time = 0

    def save_model(self, filename=None, location=None):
        """Saving the model to a file. """
        model_path = Config(
        ).params['model_path'] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            net_gen_path = f'{model_path}/Generator_{filename}'
            net_disc_path = f'{model_path}/Discriminator_{filename}'
        else:
            net_gen_path = f'{model_path}/Generator_{model_name}.pth'
            net_disc_path = f'{model_path}/Discriminator_{model_name}.pth'

        torch.save(self.generator.state_dict(), net_gen_path)
        torch.save(self.discriminator.state_dict(), net_disc_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Generator Model saved to %s.",
                         os.getpid(), net_gen_path)
            logging.info("[Server #%d] Discriminator Model saved to %s.",
                         os.getpid(), net_disc_path)
        else:
            logging.info("[Client #%d] Generator Model saved to %s.",
                         self.client_id, net_gen_path)
            logging.info("[Client #%d] Discriminator Model saved to %s.",
                         self.client_id, net_disc_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file. """
        model_path = Config(
        ).params['model_path'] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            net_gen_path = f'{model_path}/Generator_{filename}'
            net_disc_path = f'{model_path}/Discriminator_{filename}'
        else:
            net_gen_path = f'{model_path}/Generator_{model_name}.pth'
            net_disc_path = f'{model_path}/Discriminator_{model_name}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading a Generator model from %s.",
                         os.getpid(), net_gen_path)
            logging.info("[Server #%d] Loading a Discriminator model from %s.",
                         os.getpid(), net_disc_path)
        else:
            logging.info("[Client #%d] Loading a Generator model from %s.",
                         self.client_id, net_gen_path)
            logging.info("[Client #%d] Loading a Discriminator model from %s.",
                         self.client_id, net_disc_path)

        self.generator.load_state_dict(torch.load(net_gen_path))
        self.discriminator.load_state_dict(torch.load(net_disc_path))

    def train_loop(self, config, trainset, sampler, cut_layer=None) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.

        Returns:
        float: The training time.
        """
        batch_size = config['batch_size']
        log_interval = 10

        logging.info("[Client #%d] Loading the dataset.", self.client_id)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   sampler=sampler)

        self.model.to(self.device)
        self.model.train()

        # self.generator.apply(self.model.weights_init)
        # self.discriminator.apply(self.model.weights_init)

        optimizer_gen = optimizers.get_optimizer(self.generator)
        optimizer_disc = optimizers.get_optimizer(self.discriminator)

        real_label = 1.
        fake_label = 0.

        epochs = config['epochs']
        for epoch in range(1, epochs + 1):
            # Here we assume the data samples still have labels attached to them,
            # but GAN training does not need labels, so we'll just discard them
            for batch_id, (examples, _) in enumerate(train_loader):
                cur_batch_size = len(examples)
                examples = examples.to(self.device)
                label = torch.full((cur_batch_size, ),
                                   real_label,
                                   dtype=torch.float)
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
                noise = torch.randn(cur_batch_size,
                                    self.model.nz,
                                    1,
                                    1,
                                    device=self.device)
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
                label.fill_(
                    real_label)  # fake labels are real for generator cost
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
                            "Discriminator Loss: %.6f", os.getpid(), epoch,
                            epochs, batch_id, len(train_loader),
                            err_gen.data.item(), err_disc_total.data.item())
                    else:
                        logging.info(
                            "[Client #%d] Epoch: [%d/%d][%d/%d]\tGenerator Loss: %.6f\t"
                            "Discriminator Loss: %.6f", self.client_id, epoch,
                            epochs, batch_id, len(train_loader),
                            err_gen.data.item(), err_disc_total.data.item())

    def test(self, testset, sampler=None) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        return 0

    async def server_test(self, testset, sampler=None):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        return 0
