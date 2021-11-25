"""The YOLOV5 model for PyTorch."""

from yolov5.models import yolo
from yolov5.models.common import *
from yolov5.models.experimental import *

from plato.config import Config

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Model(yolo.Model):
    """The YOLOV5 model with cut layer support."""
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
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
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
                self._profile_one_layer(m, x, dt)

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    @staticmethod
    def get_model():
        """Obtaining an instance of this model provided that the name is valid."""
        if hasattr(Config().trainer, 'model_config'):
            return Model(Config().trainer.model_config,
                         Config().data.num_classes)
        else:
            return Model('yolov5s.yaml', Config().data.num_classes)
