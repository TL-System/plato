"""The YOLOV5 model for PyTorch."""
from yolov5.models import yolo
from yolov5.utils.torch_utils import time_synchronized
from yolov5.utils.loss import ComputeLoss
from yolov5.test import test
from yolov5.utils.general import one_cycle

import logging
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml

from config import Config
from datasets import coco

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class yololoss:
    """ Compute YOLOv5 losses. """
    def __init__(self, model):
        self.model = model
        self.loss = ComputeLoss(self.model)

    def __call__(self, p, targets):
        return self.loss(p, targets)


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

    def loss_criterion(self, model):
        """The loss criterion for training YOLOv5."""
        return yololoss(model)

    def train_loader(self, batch_size, trainset, cut_layer=None):
        """The train loader for training YOLOv5 using the COCO dataset."""
        return coco.Dataset.get_train_loader(batch_size, trainset, cut_layer)

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            model: The model.
            testset: The test dataset.
        """
        assert Config().data.dataset == 'COCO'
        test_loader = coco.Dataset.get_test_loader(config['batch_size'],
                                                   testset)

        results, __, __ = test('packages/yolov5/yolov5/data/coco128.yaml',
                               batch_size=config['batch_size'],
                               imgsz=640,
                               model=self,
                               single_cls=False,
                               dataloader=test_loader,
                               save_dir='',
                               verbose=False,
                               plots=False,
                               log_imgs=0,
                               compute_loss=None)
        return results[2]

    @staticmethod
    def train_model(self, config, trainset, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for YOLOv5.

        Arguments:
        rank: Required by torch.multiprocessing to spawn processes. Unused.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

        batch_size = config['batch_size']

        cuda = (self.device != 'cpu')
        nc = Config().data.num_classes  # number of classes
        names = Config().data.classes  # class names
        total_batch_size = batch_size

        with open('packages/yolov5/yolov5/data/hyp.scratch.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
        logging.info(f"Scaled weight_decay = {hyp['weight_decay']}")
        logging.info("[Client %s] Loading the dataset.", self.client_id)

        if cut_layer is None:
            train_loader = self.model.train_loader(batch_size, trainset, cut_layer)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True)


        nb = len(train_loader)
        epochs = config['epochs']

        # Sending the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        # Initializing the optimizer
        if Config().trainer.optimizer == 'Adam':
            optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logging.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        if Config().trainer.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Image sizes
        nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])

        # DP mode
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (Config().data.image_size / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.names = names
        nw = max(round(hyp['warmup_epochs'] * nb), 1000)
        scaler = amp.GradScaler(enabled=cuda)
        optimizer.zero_grad()

        # Initializing the loss criterion
        loss_criterion = self.model.loss_criterion(self.model)
        for epoch in range(1, epochs + 1):
            logging.info(('\n' + '%10s' * 8) % (
                'Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb)
            mloss = torch.zeros(4, device=self.device)  # Initializing mean losses

            for batch_id, (examples, labels) in pbar:
                ni = batch_id + nb * epoch
                examples, labels = examples.to(self.device), labels.to(
                        self.device)

                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                with amp.autocast(enabled=cuda):
                    if cut_layer is None:
                            outputs = self.model(examples)
                    else:
                            outputs = self.model.forward_from(examples, cut_layer)


                loss, loss_items = loss_criterion(outputs, labels)
                scaler.scale(loss).backward()

                # optimizer.step()
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()


                mloss = (mloss * batch_id + loss_items) / (batch_id + 1)  # update mean losses
                mem = '%.3gG' % (
                                torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                                '%g/%g' % (epoch, epochs - 1), mem, *mloss, labels.shape[0], examples.shape[-1])
                pbar.set_description(s)

            if lr_schedule is not None:
                lr_schedule.step()