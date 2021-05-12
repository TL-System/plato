"""U-Net model for image segmentation
refer from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
import torch
import torch.nn as nn
import torchvision


class Block(nn.Module):
    """Basic block for U-Net architecture"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """The Encoder is the contractive path of the U-Net architecture"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.MaxPool2d(2),
                                     Block(in_channels, out_channels))

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """The Decoder is the expansive path of the U-Net architecture"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up_sample = nn.Upsample(scale_factor=2,
                                         mode='bilinear',
                                         align_corners=True)
            self.block = Block(in_channels, out_channels, in_channels // 2)
        else:
            self.up_sample = nn.ConvTranspose2d(in_channels,
                                                in_channels // 2,
                                                kernel_size=2,
                                                stride=2)
            self.block = Block(in_channels, out_channels)

    def forward(self, x, encoder_features):

        x = self.up_sample(x)
        enc_ftrs = self.crop(encoder_features, x)
        x = torch.cat([x, enc_ftrs], dim=1)
        x = self.block(x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Model(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input = Block(n_channels, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)
        factor = 2 if bilinear else 1
        self.encoder4 = Encoder(512, 1024 // factor)
        self.decoder1 = Decoder(1024, 512 // factor, bilinear)
        self.decoder2 = Decoder(512, 256 // factor, bilinear)
        self.decoder3 = Decoder(256, 128 // factor, bilinear)
        self.decoder4 = Decoder(128, 64, bilinear)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        logits = self.output(x)
        return logits

    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        return Model()