"""The YOLOV5 model for PyTorch."""

from models.yolov5 import yolo
from config import Config
from utils.yolov5.torch_utils import time_synchronized
from utils.yolov5.datasets import LoadImagesAndLabels
from utils.yolov5.test import testmap
import logging
import os
import numpy as np

import torch
import torch.nn as nn
import wandb
from utils import optimizers
from trainers.trainer import Trainer
from utils.yolov5.loss import yololss

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Model(yolo.Model):
    """The YOLOV5 model."""
    def __init__(self, model_config, num_classes):
        super().__init__(cfg=model_config, ch=3, nc=num_classes)
        Config().params['grid_size'] = int(self.stride.max())

    def forward_to(self, x, cut_layer=4, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:

            if m.i == cut_layer:
                return x

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j]
                                    for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(
                    x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def forward_from(self, x, cut_layer=4, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.i < cut_layer:
                y.append(None)
                continue
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j]
                                    for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(
                    x, ), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name == 'yolov5'

    @staticmethod
    def get_model_from_name(model_name):
        """Obtaining an instance of this model provided that the name is valid."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        if hasattr(Config().trainer, 'model_config'):
            return Model(Config().trainer.model_config,
                         Config().data.num_classes)
        else:
            return Model('yolov5s.yaml', Config().data.num_classes)

    def is_train_process(self):
        return True

    def is_test_process(self):
        return True

    @staticmethod
    def train_process(rank,self, config, trainset, cut_layer=None):  # pylint: disable=unused-argument
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """
        run = wandb.init(project="plato",
                         group=str(config['run_id']),
                         reinit=True)

        log_interval = 10
        batch_size = config['batch_size']
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=LoadImagesAndLabels.collate_fn
                                                   )
        iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        loss_criterion = yololss(self.model)
        # Initializing the optimizer

        optimizer = optimizers.get_optimizer(self.model)
        # optimizer = self.customize_optimizer_setup(optimizer)
        # Initializing the learning rate schedule, if necessary
        if hasattr(Config().trainer, 'lr_schedule'):
            lr_schedule = optimizers.get_lr_schedule(optimizer,
                                                     iterations_per_epoch,
                                                     train_loader)
        else:
            lr_schedule = None

        for epoch in range(1, epochs + 1):
            for batch_id, (examples, labels, _, _) in enumerate(train_loader):
                examples, labels = examples.to(self.device).float() / 255.0, labels.to(
                    self.device)
                optimizer.zero_grad()

                if cut_layer is None:
                    outputs = self.model(examples)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)

                loss = loss_criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                if lr_schedule is not None:
                    lr_schedule.step()

                if config['optimizer'] == 'FedProx':
                    optimizer.params_state_update()

                if batch_id % log_interval == 0:
                    if self.client_id == 0:
                        logging.info(
                            "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                                format(os.getpid(), epoch, epochs, batch_id,
                                       len(train_loader), loss.data.item()))
                    else:
                        wandb.log({"batch loss": loss.data.item()})

                        logging.info(
                            "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}".
                                format(self.client_id, epoch, epochs, batch_id,
                                       len(train_loader), loss.data.item()))
        self.model.cpu()

        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
        Trainer.save_model(self.model,self.client_id,filename)

        run.finish()

    @staticmethod
    def test_process(rank, self, config, testset):  # pylint: disable=unused-argument
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        rank: Required by torch.multiprocessing to spawn processes. Unused.
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
        self.model.to(self.device)
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False, collate_fn=LoadImagesAndLabels.collate_fn
        )


        results, maps, times = testmap('utils/yolov5/coco128.yaml',
                                       batch_size=config['batch_size'],
                                       imgsz=640,
                                       model=self.model,
                                       single_cls=False,
                                       dataloader=test_loader,
                                       save_dir='',
                                       verbose=False,
                                       plots=False,
                                       log_imgs=0,
                                       compute_loss=None)
        accuracy = results[2]

        self.model.cpu()

        model_type = Config().trainer.model
        filename = f"{model_type}_{self.client_id}_{config['run_id']}.acc"
        Trainer.save_accuracy(accuracy, filename)