"""
Implement the trainer for self-supervised learning method.

"""

import os
import logging
import time
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers

from plato.models import ssl_monitor_register


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [
            torch.zeros_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input, ) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    """ The NTXent loss utilized by most self-supervised methods. """

    def __init__(self, batch_size, temperature, world_size=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1),
                                z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class Trainer(basic.Trainer):

    def __init__(self, model=None):
        super().__init__(model)

        # the client's personalized model
        # to perform the evaluation stage of the ssl methods
        # the client must assign its own personalized model
        #  to its trainer
        self.personalized_model = None

    def set_client_personalized_model(self, personalized_model):
        """ Setting the client's personalized model """
        self.personalized_model = personalized_model

    def loss_criterion(self, model):
        """ The loss computation. 
            Currently, we only support the NT_Xent.

            The pytorch_metric_learning provides a strong 
            support for loss criterion. However, how to use
            its NTXent is still nor clear. 
            The loss criterion will be replaced by the one 
            in pytorch_metric_learning afterward.
        """
        # define the loss computation instance
        defined_temperature = Config().trainer.temperature
        batch_size = Config().trainer.batch_size

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            z1, z2 = outputs
            criterion = NT_Xent(batch_size, defined_temperature, world_size=1)
            loss = criterion(z1, z2)
            return loss

        return loss_compute

    def save_personalized_model(self, filename=None, location=None):
        """ Saving the model to a file. """
        model_path = Config(
        ).params['model_path'] if location is None else location
        personalized_model_name = Config().trainer.personalized_model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f'{model_path}/{filename}'
        else:
            model_path = f'{model_path}/{personalized_model_name}.pth'

        if self.model_state_dict is None:
            torch.save(self.personalized_model.state_dict(), model_path)
        else:
            torch.save(self.model_state_dict, model_path)

        logging.info("[Client #%d] Personalized Model saved to %s.",
                     self.client_id, model_path)

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        model_path = Config(
        ).params['model_path'] if location is None else location
        personalized_model_name = Config().trainer.personalized_model_name

        if filename is not None:
            model_path = f'{model_path}/{filename}'
        else:
            model_path = f'{model_path}/{personalized_model_name}.pth'

        logging.info("[Client #%d] Loading a Personalized model from %s.",
                     self.client_id, model_path)

        self.personalized_model.load_state_dict(torch.load(model_path),
                                                strict=True)

    def train_loop(self, config, trainset, sampler, cut_layer):
        """ The default training loop when a custom training loop is not supplied. """
        batch_size = config['batch_size']

        tic = time.perf_counter()

        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        _train_loader = getattr(self, "train_loader", None)

        if callable(_train_loader):
            train_loader = self.train_loader(batch_size, trainset, sampler,
                                             cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=batch_size,
                                                       sampler=sampler)

        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_dynamic_optimizer = getattr(self, "get_optimizer",
                                        optimizers.get_dynamic_optimizer)
        optimizer = get_dynamic_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if 'lr_schedule' in config:
            lr_schedule = optimizers.get_dynamic_lr_schedule(
                optimizer, iterations_per_epoch, train_loader)
        else:
            lr_schedule = None

        epoch_log_interval = config['epoch_log_interval']
        batch_log_interval = config['batch_log_interval']
        epochs = config['epochs']
        epoch_loss_meter = optimizers.AverageMeter(name='Loss')
        batch_loss_meter = optimizers.AverageMeter(name='Loss')

        for epoch in range(1, epochs + 1):
            epoch_loss_meter.reset()
            # Use a default training loop
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples1, examples2 = examples
                examples1, examples2, labels = examples1.to(
                    self.device), examples2.to(self.device), labels.to(
                        self.device)

                batch_loss_meter.reset()
                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples1, examples2)
                else:
                    outputs = self.model.forward_from(examples1, examples2,
                                                      cut_layer)

                loss = loss_criterion(outputs, labels)

                if 'create_graph' in config:
                    loss.backward(create_graph=config['create_graph'])
                else:
                    loss.backward()

                optimizer.step()
                epoch_loss_meter.update(loss.data.item())
                batch_loss_meter.update(loss.data.item())

                if batch_id % batch_log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #%d] Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            os.getpid(), epoch, epochs, batch_id,
                            len(train_loader), batch_loss_meter.avg)
                    else:
                        logging.info(
                            "   [Client #%d] Contrastive Pre-train Epoch: [%d/%d][%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, epochs, batch_id,
                            len(train_loader), batch_loss_meter.avg)

            if epoch - 1 % epoch_log_interval == 0:
                logging.info(
                    "[Client #%d] Contrastive Pre-train Epoch: [%d/%d]\tLoss: %.6f",
                    self.client_id, epoch, epochs, epoch_loss_meter.avg)
            if lr_schedule is not None:
                lr_schedule.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if self.client_id != 0 and hasattr(
                    Config().clients,
                    "speed_simulation") and Config().clients.speed_simulation:
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if hasattr(Config().server,
                       'request_update') and Config().server.request_update:
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False)
                    if "monitor_trainset" in list(kwargs.keys()):
                        monitor_train_loader = torch.utils.data.DataLoader(
                            kwargs["monitor_trainset"],
                            batch_size=config['batch_size'],
                            shuffle=False)
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        sampler=sampler.get())
                    if "monitor_trainset" in list(kwargs.keys()):
                        monitor_train_loader = torch.utils.data.DataLoader(
                            kwargs["monitor_trainset"],
                            batch_size=config['batch_size'],
                            shuffle=False,
                            sampler=kwargs["monitor_trainset_sampler"].get())

                accuracy = ssl_monitor_register.get()(
                    encoder=self.model.encoder,
                    monitor_data_loader=monitor_train_loader,
                    test_data_loader=test_loader,
                    device=self.device)

        except Exception as testing_exception:
            logging.info("Monitor Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def eval_test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        self.personalized_model.to(self.device)

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            custom_test = getattr(self, "eval_test_model", None)

            if callable(custom_test):
                accuracy = self.eval_test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['pers_batch_size'],
                        shuffle=False)
                    if "eval_trainset" in list(kwargs.keys()):
                        eval_train_loader = torch.utils.data.DataLoader(
                            kwargs["eval_trainset"],
                            batch_size=config['pers_batch_size'],
                            shuffle=False)
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['pers_batch_size'],
                        shuffle=False,
                        sampler=sampler.get())
                    if "eval_trainset" in list(kwargs.keys()):
                        eval_train_loader = torch.utils.data.DataLoader(
                            kwargs["eval_trainset"],
                            batch_size=config['pers_batch_size'],
                            shuffle=False,
                            sampler=kwargs["eval_trainset_sampler"].get())

                # perform the evaluation in the downstream task
                #   i.e., the client's personal local dataset
                eval_optimizer = optimizers.get_dynamic_optimizer(
                    self.personalized_model, prefix="pers_")
                iterations_per_epoch = np.ceil(
                    len(kwargs["eval_trainset"]) /
                    Config().trainer.pers_batch_size).astype(int)
                # Initializing the learning rate schedule, if necessary

                if 'pers_lr_schedule' in config:
                    lr_schedule = optimizers.get_dynamic_lr_schedule(
                        optimizer=eval_optimizer,
                        iterations_per_epoch=iterations_per_epoch,
                        train_loader=eval_train_loader,
                        prefix="pers_")
                else:
                    lr_schedule = None

                # Initializing the loss criterion
                _eval_loss_criterion = getattr(self, "eval_loss_criterion",
                                               None)
                if callable(_eval_loss_criterion):
                    eval_loss_criterion = self.eval_loss_criterion(self.model)
                else:
                    eval_loss_criterion = torch.nn.CrossEntropyLoss()

                self.model.eval()
                self.personalized_model.train()

                epoch_log_interval = config['epoch_log_interval']
                num_eval_train_epochs = Config().trainer.pers_epochs
                epoch_loss_meter = optimizers.AverageMeter(name='Loss')

                # Start eval training
                global_progress = tqdm(range(0, num_eval_train_epochs),
                                       desc=f'Evaluating')
                for epoch in global_progress:
                    epoch_loss_meter.reset()
                    local_progress = tqdm(
                        eval_train_loader,
                        desc=f'Epoch {epoch}/{num_eval_train_epochs}',
                        disable=True)

                    for idx, (examples, labels) in enumerate(local_progress):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        eval_optimizer.zero_grad()

                        with torch.no_grad():
                            feature = self.model.encoder(examples)

                        preds = self.personalized_model(feature)

                        loss = eval_loss_criterion(preds, labels)

                        loss.backward()
                        eval_optimizer.step()
                        epoch_loss_meter.update(loss.data.item())

                        if lr_schedule is not None:
                            lr_schedule = lr_schedule.step()
                        local_progress.set_postfix({
                            'lr':
                            lr_schedule,
                            "loss":
                            epoch_loss_meter.val,
                            'loss_avg':
                            epoch_loss_meter.avg
                        })

                    logging.info(
                        "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                        self.client_id, epoch, num_eval_train_epochs,
                        epoch_loss_meter.avg)
                # perform the test phase of the eval stage
                acc_meter = optimizers.AverageMeter(name='Accuracy')

                self.personalized_model.eval()
                correct, total = 0, 0
                acc_meter.reset()
                for idx, (examples, labels) in enumerate(test_loader):
                    examples, labels = examples.to(self.device), labels.to(
                        self.device)
                    with torch.no_grad():
                        feature = self.model.encoder(examples)
                        preds = self.personalized_model(feature).argmax(dim=1)
                        correct = (preds == labels).sum().item()
                        acc_meter.update(correct / preds.shape[0])
                accuracy = acc_meter.avg
        except Exception as testing_exception:
            logging.info("Evaluation Testing on client #%d failed.",
                         self.client_id)
            raise testing_exception

        if 'max_concurrency' in config:
            self.personalized_model.cpu()
            model_type = config['personalized_model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_personalized_model(filename)

        if 'max_concurrency' in config:
            model_name = config['personalized_model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def eval_test(self, testset, sampler=None, **kwargs) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.eval_test_process,
                              args=(config, testset, sampler),
                              kwargs=kwargs)
            proc.start()
            proc.join()

            accuracy = -1
            try:
                model_name = Config().trainer.personalized_model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Evaluation testing on client #{self.client_id} failed."
                ) from error

            self.pause_training()
        else:
            accuracy = self.eval_test_process(config, testset, **kwargs)

        return accuracy