"""The YOLOV5 model for PyTorch."""
import torch
from config import Config

from trainers.trainer import Trainer

from models.yolov5 import yolo
from utils.yolov5.torch_utils import time_synchronized
from utils.yolov5.test import testmap
from utils.yolov5.loss import ComputeLoss
from datasets import coco

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class yololoss:
    # Compute losses
    def __init__(self, model):
        self.model = model
        nc = Config().data.num_classes
        nl = self.model.model[-1].nl
        import yaml
        with open('utils/yolov5/hyp.scratch.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (640 /
                       640)**2 * 3. / nl  # scale to image size and layers
        self.model.nc = nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.loss = ComputeLoss(self.model)

    def __call__(self, p, targets):
        return self.loss(p, targets)[0]


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

    def test(self, config, testset):  # pylint: disable=unused-argument
        """The testing loop for YOLOv5.

        Arguments:
            config: Configuration parameters as a dictionary.
            model: The model.
            testset: The test dataset.
        """
        assert Config().data.dataset == 'COCO'
        test_loader = coco.Dataset.get_test_loader(config['batch_size'],
                                                   testset)

        results, __, __ = testmap('utils/yolov5/coco128.yaml',
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
