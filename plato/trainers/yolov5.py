"""The YOLOV5 model for PyTorch."""
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from torch import nn, optim
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from yolov5.utils.general import (
    NCOLS,
    box_iou,
    check_dataset,
    non_max_suppression,
    one_cycle,
    scale_boxes,
    xywh2xyxy,
)
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.torch_utils import time_sync

from plato.config import Config
from plato.datasources import yolo
from plato.trainers import basic
from plato.utils import unary_encoding


class Trainer(basic.Trainer):
    """The YOLOV5 trainer."""

    def __init__(self):
        super().__init__()
        Config().params["grid_size"] = int(self.model.stride.max())

    @staticmethod
    def get_train_loader(
        batch_size,
        trainset,
        sampler,
        extract_features=False,
        cut_layer=None,
        **kwargs,
    ):
        """The train loader for training YOLOv5 using the COCO dataset or other datasets for the
        YOLOv5 model.
        """
        return yolo.DataSource.get_train_loader(
            batch_size, trainset, sampler, extract_features, cut_layer
        )

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The training loop for YOLOv5.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        """

        logging.info("[Client #%d] Setting up training parameters.", self.client_id)

        batch_size = config["batch_size"]
        total_batch_size = batch_size
        epochs = config["epochs"]

        cuda = self.device != "cpu"
        nc = Config().data.num_classes  # number of classes
        names = Config().data.classes  # class names

        with open(Config().parameters.trainer.train_params, encoding="utf-8") as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False

        nbs = 64  # nominal batch size
        accumulate = max(
            round(nbs / total_batch_size), 1
        )  # accumulate loss before optimizing
        hyp["weight_decay"] *= total_batch_size * accumulate / nbs  # scale weight_decay

        # Sending the model to the device used for training
        self.model.to(self.device)

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        # Initializing the optimizer
        if Config().trainer.optimizer == "Adam":
            optimizer = optim.Adam(
                pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
            )  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(
                pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
            )

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info(
            "[Client #%s] Optimizer groups: %g .bias, %g conv.weight, %g other",
            self.client_id,
            len(pg2),
            len(pg1),
            len(pg0),
        )
        del pg0, pg1, pg2

        if Config().trainer.linear_lr:
            lf = (
                lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]
            )  # linear
        else:
            lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

        # Image sizes
        nl = self.model.model[
            -1
        ].nl  # number of detection layers (used for scaling hyp['obj'])

        # Trainloader
        logging.info("[Client #%d] Loading the dataset.", self.client_id)
        train_loader = Trainer.get_train_loader(
            batch_size, trainset, sampler, cut_layer=self.model.cut_layer
        )
        nb = len(train_loader)

        # Model parameters
        hyp["box"] *= 3.0 / nl  # scale to layers
        hyp["cls"] *= nc / 80.0 * 3.0 / nl  # scale to classes and layers
        hyp["obj"] *= (
            (Config().data.image_size / 640) ** 2 * 3.0 / nl
        )  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.names = names

        # Start training
        nw = max(
            round(hyp["warmup_epochs"] * nb), 1000
        )  # number of warmup iterations, max(3 epochs, 1k iterations)
        last_opt_step = -1
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(self.model)  # init loss class

        for epoch in range(1, epochs + 1):
            self.model.train()
            logging.info(
                ("\n" + "%10s" * 7)
                % ("Epoch", "gpu_mem", "box", "obj", "cls", "labels", "img_size")
            )
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb)
            mloss = torch.zeros(3, device=self.device)  # mean losses
            optimizer.zero_grad()

            for i, (imgs, targets, *__) in pbar:
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(
                        1, np.interp(ni, xi, [1, nbs / batch_size]).round()
                    )
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                hyp["warmup_bias_lr"] if j == 2 else 0.0,
                                x["initial_lr"] * lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni, xi, [hyp["warmup_momentum"], hyp["momentum"]]
                            )

                # Forward
                with amp.autocast(enabled=cuda):
                    if self.model.cut_layer is None:
                        pred = self.model(imgs)
                    else:
                        pred = self.model.forward_from(imgs)

                    loss, loss_items = compute_loss(
                        pred, targets.to(self.device)
                    )  # loss scaled by batch_size

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                if ni - last_opt_step >= accumulate:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    last_opt_step = ni

                # Print
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%10s" * 2 + "%10.4g" * 5)
                    % (
                        f"{epoch}/{epochs}",
                        mem,
                        *mloss,
                        targets.shape[0],
                        imgs.shape[-1],
                    )
                )

            lr_scheduler.step()

    @staticmethod
    def process_batch(detections, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(
            detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
        )
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where(
            (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
        )  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iouv.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
        return correct

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        assert Config().data.datasource == "YOLO"
        test_loader = yolo.DataSource.get_test_loader(config["batch_size"], testset)

        device = next(self.model.parameters()).device  # get model device

        # Configure
        self.model.eval()
        with open(Config().data.data_params, encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        check_dataset(data)  # check
        nc = Config().data.num_classes  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = {
            k: v
            for k, v in enumerate(
                self.model.names
                if hasattr(self.model, "names")
                else self.model.module.names
            )
        }
        s = ("%20s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Labels",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        dt, p, r, __, mp, mr, map50, map = (
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        stats, ap = [], []
        pbar = tqdm(
            test_loader,
            desc=s,
            ncols=NCOLS,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )  # progress bar

        for __, (img, targets, paths, shapes) in enumerate(pbar):
            t1 = time_sync()
            img = img.to(device, non_blocking=True).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            __, __, height, width = img.shape  # batch size, channels, height, width
            t2 = time_sync()
            dt[0] += t2 - t1

            with torch.no_grad():
                # Run model
                if Config().algorithm.type == "mistnet":
                    logits = self.model.forward_to(img)
                    logits = logits.cpu().detach().numpy()
                    logits = unary_encoding.encode(logits)
                    logits = torch.from_numpy(logits.astype("float32"))
                    out, __ = self.model.forward_from(logits.to(device))
                else:
                    out, __ = self.model(img)

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
                    device
                )  # to pixels

                lb = []  # for autolabelling
                out = non_max_suppression(
                    out, conf_thres=0.001, iou_thres=0.6, labels=lb, multi_label=True
                )

            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                __, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                tcls,
                            )
                        )
                    continue

                # Predictions
                predn = pred.clone()
                scale_boxes(
                    img[si].shape[1:], predn[:, :4], shape, shapes[si][1]
                )  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(
                        img[si].shape[1:], tbox, shape, shapes[si][1]
                    )  # native-space labels
                    labelsn = torch.cat(
                        (labels[:, 0:1], tbox), 1
                    )  # native-space labels
                    correct = self.process_batch(predn, labelsn, iouv)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append(
                    (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
                )  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            __, __, p, r, __, ap, __ = ap_per_class(
                *stats, plot=False, save_dir="", names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        return map50

    def randomize(self, bit_array: np.ndarray, targets: np.ndarray, epsilon):
        """
        The object detection unary encoding method.
        """
        assert isinstance(bit_array, np.ndarray)

        img = unary_encoding.symmetric_unary_encoding(bit_array, 1)
        label = unary_encoding.symmetric_unary_encoding(bit_array, epsilon)
        targets_new = targets.clone().detach().numpy()

        for i in range(targets_new.shape[1]):
            box = Trainer.convert(bit_array.shape[2:], targets_new[0][i][2:])
            img[:, :, box[0] : box[2], box[1] : box[3]] = label[
                :, :, box[0] : box[2], box[1] : box[3]
            ]

        return img

    @staticmethod
    def convert(size, box):
        """Converts YOLOv5 input features.
        size: The input feature size (w, h).
        box: (xmin, xmax, ymin, ymax).
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

        return (int(x1), int(y1), int(x2), int(y2))
