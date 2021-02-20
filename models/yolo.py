"""The YOLOV5 model for PyTorch."""

from models.yolov5 import yolo
from config import Config
from utils.yolov5.torch_utils import time_synchronized
from utils.yolov5.test import testmap
from utils.yolov5.datasets import LoadImagesAndLabels

import torch
from trainers.trainer import Trainer
from utils.yolov5.loss import ComputeLoss

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)

class yololoss:
    # Compute losses
    def __init__(self, model):
        super(yololoss, self).__init__()
        # from utils.yolov5.loss import ComputeLoss
        self.model = model
        nc = Config().data.num_classes
        nl = self.model.model[-1].nl
        import yaml
        with open('utils/yolov5/hyp.scratch.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (640 / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.loss = ComputeLoss(self.model)

    def __call__(self, p, targets):
        return self.loss(p,targets)[0]

class YoloDataset(torch.utils.data.Dataset):
    """Used to prepare a feature dataset for a DataLoader in PyTorch."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label, _, _ = self.dataset[item]
        return image / 255.0, label


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


    def trainloader(self, batch_size, trainset):
        return torch.utils.data.DataLoader(YoloDataset(trainset),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_fn
                                                   )

    def loss_fun(self, model):
        return yololoss(model)


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
