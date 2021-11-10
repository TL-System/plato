"""
The training and testing loops for PyTorch.
"""
import copy
import logging
import os
import random
import multiprocessing as mp

import numpy as np
import wandb
import torch
import torch.nn as nn
import higher

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


class Trainer(basic.Trainer):
    """A federated learning trainer for personalized FL using the one-step meta-learning."""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)
        self.test_meta_personalization = False

        # the author in the paper make first-order approximation
        #   thus the D^i = ^{prime prime}_i
        self.is_consistent_data = True

        # define the optimizer for the inner iteration
        self.inner_optimizer = None
        # define the optimizer for the meta model
        self.meta_optimizer = None

    def save_specific_model(self, model, model_name, filename=None):
        """Saving the model to a file."""
        model_dir = Config().params['model_dir']

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}{filename}'
        else:
            model_path = f'{model_dir}{model_name}.pth'

        torch.save(model.state_dict(), model_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Model %s saved to %s.", os.getpid(),
                         model_name, model_path)
        else:
            logging.info("[Client #%d] Model %s saved to %s.", self.client_id,
                         model_name, model_path)

    def define_train_items(self, config):
        """ Define the necessary items used for meta training """

        local_update_steps = config['local_update_steps']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        if self.meta_optimizer is None:
            # The learning rate here is the meta learning rate (beta)
            self.meta_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=Config().trainer.meta_learning_rate,
                momentum=Config().trainer.momentum,
                weight_decay=Config().trainer.weight_decay)

        if self.inner_optimizer is None:
            self.inner_optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=Config().trainer.learning_rate,
                momentum=Config().trainer.momentum,
                weight_decay=Config().trainer.weight_decay)

        # Initializing the schedule for meta learning rate, if necessary
        if hasattr(config, 'meta_lr_schedule'):
            meta_lr_schedule = optimizers.get_lr_schedule(
                self.meta_optimizer, local_update_steps)
        else:
            meta_lr_schedule = None

        # Initializing the learning rate schedule, if necessary
        if hasattr(config, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(self.inner_optimizer,
                                                     local_update_steps)
        else:
            lr_schedule = None

        return local_update_steps, meta_lr_schedule, lr_schedule

    def generate_train_batch_range(self, dataset_loader, local_update_steps):
        """ Generate the separated data pools for selecting D^i, D^{prime_i},
            D^{prime prime}_i for meta-learning """

        # to ensure the three batches are always independent from each other
        #   we split the dataset into three parts based on the
        #   batches required in the training process
        total_num_batches = int(len(dataset_loader))

        num_splits = 2  # default two independent subsets
        if self.is_consistent_data:
            num_splits = 2
        else:
            num_splits = 3

        num_chunks = int(total_num_batches / num_splits)

        start_chunk_idx = random.randint(0, num_chunks)
        end_batch_idx = start_chunk_idx + local_update_steps * num_splits

        batches_range = np.arange(start_chunk_idx, end_batch_idx, 1)

        return batches_range, num_splits

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload."""

        if 'use_wandb' in config:
            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            # we always initialize the inner optimizer
            self.inner_optimizer = None
            self.define_train_items(config)

            # currently, we only support the same batch size for
            #   all separated sub datasets
            batch_size = config['batch_size']

            logging.info("[Client #%d] Loading the dataset for training.",
                         self.client_id)

            train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       shuffle=False,
                                                       batch_size=batch_size,
                                                       sampler=sampler.get())

            local_update_steps, meta_lr_schedule, lr_schedule = self.define_train_items(
                config)

            # initialize the loss criterion
            loss_criterion = nn.CrossEntropyLoss()

            utilized_batches_range, _ = self.generate_train_batch_range(
                train_loader, local_update_steps)

            batch_idx_flag = 0
            inner_iter_id = 0
            train_loader_iter = iter(train_loader)
            meta_train_accuracy = 0
            for _ in range(len(train_loader)):

                if batch_idx_flag == 0:
                    _ = next(train_loader_iter)
                    inner_iter_id += 1

                if inner_iter_id in utilized_batches_range:
                    batch_idx_flag = 1

                    # this is actually the D^{prime prime}_i in the algorithm
                    ## pers_update_data_batch = adap_batch_data
                    inner_iter_id += len(utilized_batches_range)

                    self.model.zero_grad()
                    meta_loss = torch.tensor(0., device=self.device)
                    with higher.innerloop_ctx(
                            self.model,
                            self.inner_optimizer,
                            track_higher_grads=False) as (fmodel, diffopt):
                        # this is actually the D^i in the algorithm
                        adap_batch_data = next(train_loader_iter)

                        adap_batch_samples = adap_batch_data[0].to(self.device)
                        adap_batch_labels = adap_batch_data[1].to(self.device)
                        adap_logits = fmodel(adap_batch_samples)
                        adap_loss = loss_criterion(adap_logits,
                                                   adap_batch_labels)

                        diffopt.step(adap_loss)
                        if lr_schedule is not None:
                            lr_schedule.step()
                        # perform one meta-train step
                        # this is actually the D^{prime_i} in the algorithm
                        meta_batch_data = next(train_loader_iter)
                        meta_batch_samples = meta_batch_data[0].to(self.device)
                        meta_batch_labels = meta_batch_data[1].to(self.device)
                        meta_logits = fmodel(meta_batch_samples)
                        meta_loss = loss_criterion(meta_logits,
                                                   meta_batch_labels)
                        meta_train_accuracy += get_accuracy(
                            logits=meta_logits, targets=meta_batch_labels)
                    meta_loss.backward()
                    self.meta_optimizer.step()
                    if meta_lr_schedule is not None:
                        meta_lr_schedule.step()

                # stop the iteration if the local train is finished
                if inner_iter_id > utilized_batches_range[-1]:
                    break
            ave_accu = float(meta_train_accuracy / local_update_steps)
            logging.info(
                "Training on client #%d generates %.5f meta accuracy.",
                self.client_id, ave_accu)

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()

    def test_process(self, config, testset, sampler):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: sampler for the testset
        """
        if not self.test_meta_personalization:
            self.model.to(self.device)
            self.model.eval()

        try:
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['batch_size'],
                shuffle=False,
                sampler=sampler.get())
            # Test its personalized model during personalization test
            if self.test_meta_personalization:
                logging.info("[Client #%d] meta Personalizing its model.",
                             self.client_id)
                # Generate a training set for personalization
                # by randomly choose one batch from test set
                random_batch_id = random.randint(0, len(test_loader) - 1)
                # initialize the loss criterion
                loss_criterion = nn.CrossEntropyLoss()
                meta_personalized_model, accuracy = self.perform_meta_learning_test(
                    initial_model=self.model,
                    test_loader=test_loader,
                    loss_criterion=loss_criterion,
                    adaptive_batch_idx=random_batch_id)
                # save the meta personalzied model once the test finished
                meta_personalized_model.cpu()
                model_type = "meat_personalized_model"
                filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
                self.save_specific_model(model=meta_personalized_model,
                                         model_name=model_type,
                                         filename=filename)

            # Directly test the trained global model on the local dataset
            else:
                accuracy = self.operate_test_model(model=self.model,
                                                   test_loader=test_loader,
                                                   masked_batches_idx=[])

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        if not self.test_meta_personalization:
            self.model.cpu()
        else:
            logging.info("[Client #%d] Finished personalization test.",
                         self.client_id)

        if 'max_concurrency' in config:
            accuracy = accuracy.item()

            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def perform_meta_learning_test(self, initial_model, test_loader,
                                   loss_criterion, adaptive_batch_idx):
        """ Perform the meta-learning test by performing one-step of SGB
            to update the initial model to the personalized model """
        meta_personalized_model = copy.deepcopy(initial_model)
        meta_personalized_model.to(self.device)
        meta_personalized_model.train()

        inner_optimizer = torch.optim.SGD(
            meta_personalized_model.parameters(),
            lr=Config().trainer.learning_rate,
            momentum=Config().trainer.momentum,
            weight_decay=Config().trainer.weight_decay)

        adap_batch = None
        for batch_id, (examples, labels) in enumerate(test_loader):
            if batch_id == adaptive_batch_idx:
                adap_batch = (examples, labels)
                break

        with higher.innerloop_ctx(meta_personalized_model,
                                  inner_optimizer,
                                  track_higher_grads=False) as (fnet, diffopt):

            # we perform one-step of meta update
            (examples, labels) = adap_batch
            spt_logits = fnet(examples)
            spt_loss = loss_criterion(spt_logits, labels)
            diffopt.step(spt_loss)

        test_accuracy = self.operate_test_model(
            model=meta_personalized_model,
            test_loader=test_loader,
            masked_batches_idx=[adaptive_batch_idx])

        return meta_personalized_model, test_accuracy

    def perform_local_personalization_test(self, trainset, testset, sampler):
        """ Perform the local personalization by training a local model based
            only on the client's local dataset """
        local_pers_personalized_model = copy.deepcopy(self.model)
        local_pers_personalized_model.to(self.device)

        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        def weight_reset(sub_module):
            if isinstance(sub_module, nn.Conv2d) or isinstance(
                    sub_module, nn.Linear):
                sub_module.reset_parameters()

        local_pers_personalized_model.apply(weight_reset)

        local_pers_personalized_model.train()

        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=Config().trainer.batch_size,
            sampler=sampler.get())
        # initialize the loss criterion
        loss_criterion = nn.CrossEntropyLoss()
        inner_optimizer = torch.optim.SGD(
            local_pers_personalized_model.parameters(),
            lr=Config().trainer.learning_rate,
            momentum=Config().trainer.momentum,
            weight_decay=Config().trainer.weight_decay)

        for _ in range(Config().trainer.local_personalization_epoches):

            for _, (examples, labels) in enumerate(train_loader):

                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                inner_optimizer.zero_grad()

                logits = local_pers_personalized_model(examples)

                loss = loss_criterion(logits, labels)

                loss.backward()

                inner_optimizer.step()

        # save the local personalzied model
        local_pers_personalized_model.cpu()
        model_type = "local_personalized_model"
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
        self.save_specific_model(model=local_pers_personalized_model,
                                 model_name=model_type,
                                 filename=filename)

        local_pers_personalized_model.eval()
        test_loader = torch.utils.data.DataLoader(
            dataset=testset,
            shuffle=False,
            batch_size=Config().trainer.batch_size,
            sampler=sampler.get())

        local_pers_accuracy = self.operate_test_model(
            model=local_pers_personalized_model,
            test_loader=test_loader,
            masked_batches_idx=[])

        return local_pers_accuracy

    def operate_test_model(self, model, test_loader, masked_batches_idx):
        """ Test the initial model received from the server directly """
        model.eval()
        accuracy = torch.tensor(0., device=self.device)
        with torch.no_grad():
            for batch_id, (examples, labels) in enumerate(test_loader):
                if batch_id in masked_batches_idx:
                    continue

                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                test_logits = model(examples)

                accuracy += get_accuracy(test_logits, labels)

        accuracy.div_(len(test_loader) - len(masked_batches_idx))

        return accuracy

    def test(self, testset, sampler) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            self.start_training()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.test_process,
                              args=(config, testset, sampler))
            proc.start()
            proc.join()

            try:
                model_name = Config().trainer.model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Testing on client #{self.client_id} failed.") from error

            self.pause_training()
        else:
            accuracy = self.test_process(config, testset, sampler)

        return accuracy
