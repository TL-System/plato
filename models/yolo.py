"""The YOLOV5 model for PyTorch."""
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml

from yolov5.models import yolo
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.datasets import LoadImagesAndLabels
from yolov5.utils.general import check_dataset, box_iou, non_max_suppression, scale_coords, xywh2xyxy, one_cycle
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.torch_utils import time_synchronized
from utils import unary_encoding

from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from config import Config
from datasources import coco


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

    def train_loader(self,
                     batch_size,
                     trainset,
                     extract_features=False,
                     cut_layer=None):
        """The train loader for training YOLOv5 using the COCO dataset."""
        return coco.DataSource.get_train_loader(batch_size, trainset,
                                                extract_features, cut_layer)

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            model: The model.
            testset: The test dataset.
        """
        assert Config().data.dataset == 'COCO'
        test_loader = coco.Dataset.get_test_loader(config['batch_size'], testset)

        device = next(self.parameters()).device  # get model device

        # Configure
        self.eval()
        with open(Config().data.data_params) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        check_dataset(data)  # check
        nc = Config().data.num_classes  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = {k: v for k, v in enumerate(self.names if hasattr(self, 'names')
                                            else self.module.names)}
        s = ('%20s' + '%12s' * 6) % \
            ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []

        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(test_loader, desc=s)):
            img = img.to(device, non_blocking=True).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                if Config().algorithm.type == 'mistnet':
                    logits = self.forward_to(img)
                    logits = logits.cpu().detach().numpy()
                    logits = unary_encoding.encode(logits)
                    logits = torch.from_numpy(logits.astype('float32'))
                    out, train_out = self.forward_from(logits.to(device))
                else:
                    out, train_out = self(img)

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = []  # for autolabelling
                out = non_max_suppression(out, conf_thres=0.001, iou_thres=0.6,
                                          labels=lb, multi_label=True)

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
                scale_coords(img[si].shape[1:], predn[:, :4],
                             shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox,
                                 shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir='', names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        return map50

    def train_model(self, trainer, config, trainset, cut_layer=None):  # pylint: disable=unused-argument
        """The training loop for YOLOv5.

        Arguments:
        trainer: The Trainer instance.
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

        logging.info("[Client #%s] Setting up training parameters.",
                     trainer.client_id)

        batch_size = config['batch_size']
        total_batch_size = batch_size
        epochs = config['epochs']

        cuda = (trainer.device != 'cpu')
        nc = Config().data.num_classes  # number of classes
        names = Config().data.classes  # class names

        with open(Config().trainer.train_params) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size),
                         1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        # Sending the model to the device used for training
        self.to(trainer.device)

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.named_modules():
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
            trainer.client_id, len(pg2), len(pg1), len(pg0))
        del pg0, pg1, pg2

        if Config().trainer.linear_lr:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp[
                'lrf']  # linear
        else:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Image sizes
        nl = self.model[
            -1].nl  # number of detection layers (used for scaling hyp['obj'])

        # Trainloader
        logging.info("[Client #%s] Loading the dataset.", trainer.client_id)
        train_loader = self.train_loader(batch_size,
                                         trainset,
                                         cut_layer=cut_layer)
        nb = len(train_loader)

        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (Config().data.image_size /
                       640)**2 * 3. / nl  # scale to image size and layers
        self.nc = nc  # attach number of classes to model
        self.hyp = hyp  # attach hyperparameters to model
        self.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.names = names

        # Start training
        nw = max(
            round(hyp['warmup_epochs'] * nb),
            1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(self)

        for epoch in range(1, epochs + 1):
            self.train()
            logging.info(
                ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls',
                                       'total', 'targets', 'img_size'))
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb)
            mloss = torch.zeros(
                4, device=trainer.device)  # Initializing mean losses
            optimizer.zero_grad()

            for i, (imgs, targets, *__) in pbar:
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs, targets = imgs.to(trainer.device), targets.to(
                    trainer.device)

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
                        pred = self(imgs)
                    else:
                        pred = self.forward_from(imgs, cut_layer)

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
