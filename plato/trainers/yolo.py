"""The YOLOV5 model for PyTorch."""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml

from plato.config import Config
from plato.datasources import yolo
from plato.trainers import basic
from plato.utils import unary_encoding
from torch.cuda import amp
from tqdm import tqdm
from yolov5.utils.general import (box_iou, check_dataset, one_cycle,
                                  non_max_suppression, scale_coords, xywh2xyxy)
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import ap_per_class


class Trainer(basic.Trainer):
    """The YOLOV5 trainer."""
    def __init__(self):
        super().__init__()
        Config().params['grid_size'] = int(self.model.stride.max())

    def train_loader(self,
                     batch_size,
                     trainset,
                     sampler,
                     extract_features=False,
                     cut_layer=None):
        """The train loader for training YOLOv5 using the COCO dataset or other datasets for the
           YOLOv5 model.
        """
        return yolo.DataSource.get_train_loader(batch_size, trainset, sampler,
                                                extract_features, cut_layer)

    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for YOLOv5.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

        logging.info("[Client #%d] Setting up training parameters.",
                     self.client_id)

        batch_size = config['batch_size']
        total_batch_size = batch_size
        epochs = config['epochs']

        cuda = (self.device != 'cpu')
        nc = Config().data.num_classes  # number of classes
        names = Config().data.classes  # class names

        with open(Config().trainer.train_params) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size),
                         1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        # Sending the model to the device used for training
        self.model.to(self.device)

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
            optimizer = optim.Adam(pg0,
                                   lr=hyp['lr0'],
                                   betas=(hyp['momentum'],
                                          0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0,
                                  lr=hyp['lr0'],
                                  momentum=hyp['momentum'],
                                  nesterov=True)

        optimizer.add_param_group({
            'params': pg1,
            'weight_decay': hyp['weight_decay']
        })  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logging.info(
            '[Client %s] Optimizer groups: %g .bias, %g conv.weight, %g other',
            self.client_id, len(pg2), len(pg1), len(pg0))
        del pg0, pg1, pg2

        if Config().trainer.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp[
                'lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Image sizes
        nl = self.model.model[
            -1].nl  # number of detection layers (used for scaling hyp['obj'])

        # Trainloader
        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        train_loader = self.train_loader(batch_size,
                                         trainset,
                                         sampler,
                                         cut_layer=cut_layer)
        nb = len(train_loader)

        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (Config().data.image_size /
                       640)**2 * 3. / nl  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.names = names

        # Start training
        nw = max(
            round(hyp['warmup_epochs'] * nb),
            1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(self.model)

        for epoch in range(1, epochs + 1):
            self.model.train()
            logging.info(
                ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls',
                                       'total', 'targets', 'img_size'))
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb)
            mloss = torch.zeros(4,
                                device=self.device)  # Initializing mean losses
            optimizer.zero_grad()

            for i, (imgs, targets, *__) in pbar:
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(
                        1,
                        np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [
                            hyp['warmup_bias_lr'] if j == 2 else 0.0,
                            x['initial_lr'] * lf(epoch)
                        ])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(
                                ni, xi,
                                [hyp['warmup_momentum'], hyp['momentum']])

                # Forward
                with amp.autocast(enabled=cuda):
                    if cut_layer is None:
                        pred = self.model(imgs)
                    else:
                        pred = self.model.forward_from(imgs, cut_layer)

                    loss, loss_items = compute_loss(
                        pred, targets)  # loss scaled by batch_size

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()

                # Print
                mloss = (mloss * i + loss_items) / (i + 1
                                                    )  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9
                                 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 +
                     '%10.4g' * 6) % ('%g/%g' % (epoch, epochs), mem, *mloss,
                                      targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            lr_schedule.step()

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        assert Config().data.datasource == 'YOLO'
        test_loader = yolo.DataSource.get_test_loader(config['batch_size'],
                                                      testset)

        device = next(self.model.parameters()).device  # get model device

        # Configure
        self.model.eval()
        with open(Config().data.data_params) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        check_dataset(data)  # check
        nc = Config().data.num_classes  # number of classes
        iouv = torch.linspace(0.5, 0.95,
                              10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = {
            k: v
            for k, v in enumerate(self.model.names if hasattr(
                self.model, 'names') else self.model.module.names)
        }
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R',
                                     'mAP@.5', 'mAP@.5:.95')
        mp, map50 = 0., 0.
        stats, ap = [], []

        for __, (img, targets, __,
                 shapes) in enumerate(tqdm(test_loader, desc=s)):
            img = img.to(device, non_blocking=True).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            __, __, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                if Config().algorithm.type == 'mistnet':
                    logits = self.model.forward_to(img)
                    logits = logits.cpu().detach().numpy()
                    logits = unary_encoding.encode(logits)
                    logits = torch.from_numpy(logits.astype('float32'))
                    out, __ = self.model.forward_from(logits.to(device))
                else:
                    out, __ = self.model(img)

                # Run NMS
                targets[:,
                        2:] *= torch.Tensor([width, height, width,
                                             height]).to(device)  # to pixels
                lb = []  # for autolabelling
                out = non_max_suppression(out,
                                          conf_thres=0.001,
                                          iou_thres=0.6,
                                          labels=lb,
                                          multi_label=True)

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                      torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0],
                             shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0],
                                      niou,
                                      dtype=torch.bool,
                                      device=device)

                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0],
                                 shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(
                            -1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(
                            -1)  # prediction indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(
                                1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[
                                        j] > iouv  # iou_thres is 1xn
                                    if len(
                                            detected
                                    ) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append(
                    (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, __, __ = ap_per_class(*stats,
                                            plot=False,
                                            save_dir='',
                                            names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64),
                             minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        return map50

    def randomize(self, bit_array: np.ndarray, targets: np.ndarray, epsilon):
        """
        The object detection unary encoding method.
        """
        assert isinstance(bit_array, np.ndarray)
        img = unary_encoding.symmetric_unary_encoding(bit_array, 1)
        label = unary_encoding.symmetric_unary_encoding(bit_array, epsilon)
        targets_new = targets.clone().detach()
        targets_new = targets_new.detach().numpy()
        for i in range(targets_new.shape[1]):
            box = self.convert(bit_array.shape[2:], targets_new[0][i][2:])
            img[:, :, box[0]:box[2],
                box[1]:box[3]] = label[:, :, box[0]:box[2], box[1]:box[3]]
        return img

    def convert(self, size, box):
        """The convert for YOLOv5.
              Arguments:
                  size: Input feature size(w,h)
                  box:(xmin,xmax,ymin,ymax).
              """
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x1 = max(x - 0.5 * w - 3, 0)
        x2 = min(x + 0.5 * w + 3, size[0])
        y1 = max(y - 0.5 * h - 3, 0)
        y2 = min(y + 0.5 * h + 3, size[1])

        x1 = round(x1 * size[0])
        x2 = round(x2 * size[0])
        y1 = round(y1 * size[1])
        y2 = round(y2 * size[1])

        return (x1, y1, x2, y2)
