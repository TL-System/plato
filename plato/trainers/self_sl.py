"""
Implement the trainer for self-supervised learning method.

"""

import os
import logging
import time
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

import torch.distributed as dist
from tqdm import tqdm
import pandas as pd

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers

from plato.models import ssl_monitor_register


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    # pylint: disable=abstract-method
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, input_tn):
        ctx.save_for_backward(input_tn)
        output = [
            torch.zeros_like(input_tn) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input_tn)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input_tn, ) = ctx.saved_tensors
        grad_out = torch.zeros_like(input_tn)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NTXent(nn.Module):
    """ The NTXent loss utilized by most self-supervised methods.

        Note: here can be important issue existed in this implementation
        of NT_Xent as:
        the NT_Xent loss utilized by the SimCLR method set the defined batch_size
        as the parameter. However, at the end of one epoch, the left samples may smaller than
        the batch_size. This makes the #loaded samples != batch_size.
        Working on criterion that is defined with batch_size but receives loaded
        samples whose size is smaller than the batch size may causes problems.
        drop_last = True can alleviate this issue.
        Currently drop_last is default to be False in Plato.
        Under this case, to avoid this issue, we need to set:
        partition_size / batch_size = integar
        partition_size / pers_batch_size = integar

    """

    def __init__(self, batch_size, temperature, world_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        """ Mask out the correlated samples. """
        batch_size, world_size = self.batch_size, self.world_size
        collected_samples = 2 * batch_size * world_size
        mask = torch.ones((collected_samples, collected_samples), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017),
        we treat the other 2(N - 1) augmented examples within
        a minibatch as negative examples.
        """
        collected_samples = 2 * self.batch_size * self.world_size

        collected_z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            collected_z = torch.cat(GatherLayer.apply(collected_z), dim=0)

        sim = self.similarity_f(collected_z.unsqueeze(1),
                                collected_z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU
        # gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i),
                                     dim=0).reshape(collected_samples, 1)

        negative_samples = sim[self.mask].reshape(collected_samples, -1)

        labels = torch.zeros(collected_samples).to(
            positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= collected_samples
        return loss


class Trainer(basic.Trainer):
    """ A federated learning trainer for self-supervised models. """

    def __init__(self, model=None):
        super().__init__(model)

        # the client's personalized model
        # to perform the evaluation stage of the ssl methods
        # the client must assign its own personalized model
        #  to its trainer
        self.personalized_model = None
        self.personalized_model_state_dict = None

    def set_client_personalized_model(self, personalized_model):
        """ Setting the client's personalized model """
        self.personalized_model = personalized_model

    @staticmethod
    def loss_criterion(model):
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
        criterion = NTXent(batch_size, defined_temperature, world_size=1)

        # currently, the loss computation only supports the one-GPU learning.
        def loss_compute(outputs, labels):
            """ A wrapper for loss computation.

                Maintain labels here for potential usage.
            """
            encoded_z1, encoded_z2 = outputs
            loss = criterion(encoded_z1, encoded_z2)
            return loss

        return loss_compute

    @staticmethod
    def process_save_path(filename, location, work_model_name,
                          desired_extenstion):
        """ Process the input arguments to obtain the final saving path. """
        # default saving everything to the model path
        model_path = Config(
        ).params['model_path'] if location is None else location
        # set the model_type to
        #  - "model_name" to obtain the global model's name
        #  - "personalized_model_name" to obtain the personalized model's name
        #  - or any other model you defined and want to process
        model_name = getattr(Config().trainer, work_model_name)
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            to_save_path = f'{model_path}/{filename}'
        else:
            to_save_path = f'{model_path}/{model_name}'

        # check the filr extension
        save_prefix, save_extension = os.path.splitext(to_save_path)
        # the save file must contain a 'pth' as its extension
        if save_extension != desired_extenstion:
            to_save_path = save_prefix + desired_extenstion

        return to_save_path

    def save_personalized_model(self, filename=None, location=None):
        """ Saving the model to a file. """
        # process the arguments to obtain the to save path
        to_save_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth")

        if self.personalized_model_state_dict is None:
            torch.save(self.personalized_model.state_dict(), to_save_path)
        else:
            torch.save(self.personalized_model_state_dict, to_save_path)

        logging.info("[Client #%d] Personalized Model saved to %s.",
                     self.client_id, to_save_path)

    def load_personalized_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        # process the arguments to obtain the to save path
        load_from_path = self.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".pth")

        logging.info("[Client #%d] Loading a Personalized model from %s.",
                     self.client_id, load_from_path)

        self.personalized_model.load_state_dict(torch.load(load_from_path),
                                                strict=True)

    @staticmethod
    def save_personalized_accuracy(accuracy,
                                   round=None,
                                   accuracy_type="monitor_accuracy",
                                   filename=None,
                                   location=None):
        """Saving the test accuracy to a file."""
        to_save_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".csv")

        current_round = round if round is not None else 0
        acc_dataframe = pd.DataFrame(
            {
                "round": current_round,
                accuracy_type: accuracy
            }, index=[0])

        is_use_header = True if not os.path.exists(
            to_save_accuracy_path) else False
        acc_dataframe.to_csv(to_save_accuracy_path,
                             index=False,
                             mode='a',
                             header=is_use_header)

    @staticmethod
    def load_personalized_accuracy(round=None,
                                   accuracy_type="monitor_accuracy",
                                   filename=None,
                                   location=None):
        """Loading the test accuracy from a file."""
        to_load_accuracy_path = Trainer.process_save_path(
            filename,
            location,
            work_model_name="personalized_model_name",
            desired_extenstion=".acc")

        loaded_rounds_accuracy = pd.read_csv(to_load_accuracy_path)
        if round is None:
            # default use the last row
            desired_row = loaded_rounds_accuracy.iloc[-1]
        else:
            desired_row = loaded_rounds_accuracy.loc[
                loaded_rounds_accuracy['round'] == round]
            desired_row = loaded_rounds_accuracy.iloc[-1]

        accuracy = desired_row[accuracy_type]

        return accuracy

    def train_loop(
        self,
        config,
        trainset,
        sampler,
        cut_layer,
        **kwargs,
    ):
        """ The default training loop when a custom training loop is not supplied.

        Note:
            This is the training stage of self-supervised learning (ssl). It is responsible
        for performing the contrastive learning process based on the trainset to train
        a encoder in the unsupervised manner. Then, this trained encoder is desired to
        use the strong backbone by downstream tasks to solve the objectives effectively.
            Therefore, the train loop here utilize the
            - trainset with one specific transform (contrastive data augmentation)
            - self.model, the ssl method to be trained.

        """
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
                            "   [Client #%d] Contrastive Pre-train Epoch: \
                            [%d/%d][%d/%d]\tLoss: %.6f",
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

        Note:
            This function performs the test process of the self-supervised learning.
        Thus, it aims to measure the quanlity of the learned representation. Generally,
        the monitors, cluster algorithms, will be used to make a classification within
        the test dataset. The general pipeline is:
            trainset -> pre-trained ssl encoder -> extracted samples' features.
            extracted samples' features -> knn -> clusters.
            testset -> trained knn -> classification.
            Therefore, the monitor process here utilize the
            - trainset with general transform
            - testset with general transform
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
                    if "monitor_trainset" in kwargs:
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
                    if "monitor_trainset" in kwargs:
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
            # first save the monitor to the results path
            results_path = Config().params['result_path']
            # the unique name set in the config file
            # to save the results
            unique_name = config['unique_name']

            save_location = os.path.join(results_path, unique_name)
            current_round = kwargs['current_round']
            filename = f"client_{self.client_id}_monitor.csv"

            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_accuracy(accuracy,
                                            round=current_round,
                                            accuracy_type="monitor_accuracy",
                                            filename=filename,
                                            location=save_location)
            # save current accuracy directly for the latter usage
            # in the test(...)
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def eval_test_process(self, config, testset, sampler=None, **kwargs):
        """ The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.

        Note:
            This is second stage of evaluation in general self-supervised learning
        methods. In this stage, the learned representation will be used for downstream
        tasks, such as image classification. The pipeline is:
            task_input -> pretrained ssl_encoder -> representation -> task_solver -> task_loss.

            But in the federated learning domain, each client perform this stage on its
        local data. The main target is to train the personalized model to complete its
        own task. Thus, the task_solver mentioned above is the personalized_model.

            By the way, the upper mentioned 'pretrained ssl_encoder' is the self.model
        in the federated learning implementation. As this is the only model shared among
        clients.
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
                    if "eval_trainset" in kwargs:
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
                    if "eval_trainset" in kwargs:
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

                epoch_log_interval = config['pers_epoch_log_interval']
                num_eval_train_epochs = Config().trainer.pers_epochs
                epoch_loss_meter = optimizers.AverageMeter(name='Loss')

                # Start eval training
                global_progress = tqdm(range(0, num_eval_train_epochs),
                                       desc='Evaluating')
                for epoch in global_progress:
                    epoch_loss_meter.reset()
                    local_progress = tqdm(
                        eval_train_loader,
                        desc=f'Epoch {epoch}/{num_eval_train_epochs}',
                        disable=True)

                    for _, (examples, labels) in enumerate(local_progress):
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

                    if epoch % epoch_log_interval == 0:
                        logging.info(
                            "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                            self.client_id, epoch, num_eval_train_epochs,
                            epoch_loss_meter.avg)
                # perform the test phase of the eval stage
                acc_meter = optimizers.AverageMeter(name='Accuracy')

                self.personalized_model.eval()
                correct = 0
                acc_meter.reset()
                for _, (examples, labels) in enumerate(test_loader):
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

        # saving the personalized model for current round
        # to the dir of this client
        if 'max_concurrency' in config:
            self.personalized_model.cpu()
            model_path = Config().params['checkpoint_path']
            # the unique name set in the config file
            # to save the results
            unique_name = config['unique_name']

            save_location = os.path.join(model_path, unique_name,
                                         "client_" + str(self.client_id))

            current_round = kwargs['current_round']
            filename = f"Round_{current_round}_personalization.pth"

            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_model(filename=filename,
                                         location=save_location)

        if 'max_concurrency' in config:
            # save the personaliation accuracy to the results dir
            results_path = Config().params['result_path']
            # the unique name set in the config file
            # to save the results
            unique_name = config['unique_name']

            save_location = os.path.join(model_path, unique_name)

            current_round = kwargs['current_round']
            filename = f"client_{self.client_id}_personalization.csv"
            os.makedirs(save_location, exist_ok=True)
            self.save_personalized_accuracy(
                accuracy,
                round=current_round,
                accuracy_type="personalization_accuracy",
                filename=filename,
                location=save_location)

            # save the accuracy directly for latter usage
            # in the eval_test(...)
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


        Note:
            It performs 'eval_test_process' that is responsible for evaluating
        the learned representation of ssl.
            Under the terminologies of federated learning, this part is to train
        the personalized model.
            However, we still call it 'eval_test' just to align the learning stage
        with that in self-supervised learning (ssl), i.e., evaluation stage.

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
