"""
Implement the trainer for the SimCLR method.

"""

import torch.nn as nn

from torchvision.models import resnet50

from plato.config import Config

backbone_mapper = {"resnet50": resnet50}


class projection_MLP(nn.Module):

    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):

    def __init__(self):
        super().__init__()
        # use resnent50 by default
        backbone_name = "resnet50"
        if hasattr(Config.trainer, "backbone_name"):
            backbone_name = Config.trainer.backbone_name

        backbone = backbone_mapper[backbone_name]()

        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return z1, z2
