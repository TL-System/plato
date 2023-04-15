"""Data augmentations as in Amin's (https://github.com/AminJun) inversion implementation
[See model-free inversion at https://arxiv.org/abs/2201.12961]
."""
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class Jitter(torch.nn.Module):
    def __init__(self, lim=32, **kwargs):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = torch.randint(-self.lim, self.lim, (1,))
        off2 = torch.randint(-self.lim, self.lim, (1,))
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


class Focus(torch.nn.Module):
    def __init__(self, size=224, std=1.0, **kwargs):
        super().__init__()
        self.size = size
        self.std = std

    def forward(self, img: torch.tensor) -> torch.tensor:
        pert = (torch.rand(2) * 2 - 1) * self.std
        w, h = img.shape[-2:]
        x = (pert[0] + w // 2 - self.size // 2).long().clamp(min=0, max=w - self.size)
        y = (pert[1] + h // 2 - self.size // 2).long().clamp(min=0, max=h - self.size)
        return img[:, :, x : x + self.size, y : y + self.size]


class Zoom(torch.nn.Module):
    def __init__(self, out_size=224, **kwargs):
        super().__init__()
        self.up = torch.nn.Upsample(size=(out_size, out_size), mode="bilinear", align_corners=False)

    def forward(self, img: torch.tensor) -> torch.tensor:
        return self.up(img)


class CenterZoom(torch.nn.Module):
    def __init__(self, initial_fov=32, out_size=224, **kwargs):
        super().__init__()
        self.fov = initial_fov
        self.out_size = out_size

    def forward(self, img: torch.tensor) -> torch.tensor:
        """Cut out a part of size fov x fov from the center of the image and zoom it to max."""
        w, h = img.shape[-2:]
        wh, hh = self.fov, self.fov
        w0, h0 = (w - wh) // 2, (h - hh) // 2
        img = img[:, :, w0 : w0 + wh, h0 : h0 + hh]
        return F.interpolate(img, size=self.out_size, mode="bilinear", align_corners=False)


class Flip(torch.nn.Module):
    def __init__(self, p=0.5, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        return torch.flip(x, dims=(3,)) if torch.rand(1,) < self.p else x

    def update(self, *args, **kwargs):
        pass


class ColorJitter(torch.nn.Module):
    def __init__(self, batch_size=1, shuffle_every=False, mean=0.0, std=1.0, **kwargs):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.shuffled = False
        self.shuffle_every = shuffle_every

    def shuffle(self, batch_size=1, dtype=torch.float, device=torch.device("cpu")):
        if self.shuffle_every or not self.shuffled:
            shape = (batch_size, 3, 1, 1)
            self.mean = (torch.rand(shape, dtype=dtype, device=device) - 0.5) * 2 * self.mean_p
            self.std = ((torch.rand(shape, dtype=dtype, device=device) - 0.5) * 2 * self.std_p).exp()

            self.shuffled = True

    def forward(self, img):
        self.shuffle(batch_size=img.shape[0], dtype=img.dtype, device=img.device)
        return (img - self.mean) / self.std


class MedianPool2d(torch.nn.Module):
    """Median pool (usable as median filter when stride=1) module.
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean

    This is code for median pooling from https://gist.github.com/rwightman.
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True, **kwargs):
        """Initialize with kernel_size, stride, padding."""
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.

    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(
        self, shift=8, fliplr=False, flipud=False, mode="bilinear", padding="reflection", align=False, **kwargs
    ):
        """Args: source and target size."""
        super().__init__()
        self.shift = shift
        self.fliplr = fliplr
        self.flipud = flipud

        self.padding = padding
        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size, dtype=torch.float, device=torch.device("cpu")):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size, device=device).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.build_grid(x.shape[2], x.shape[2], device=x.device, dtype=x.dtype).repeat(x.shape[0], 1, 1, 1)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        delta = self.shift / (x.shape[2] - 1)  # Shifts are with a magnitude in [0, 1]
        x_shift = (randgen[:, 0] - 0.5) * 2 * delta  # and shift within [-1, 1]
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift[..., None, None].expand(-1, grid.shape[1], grid.shape[2])
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift[..., None, None].expand(-1, grid.shape[1], grid.shape[2])

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid

    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        if self.padding == "circular":
            grid_shifted = (grid_shifted + 1) % 1 - 1
            padding = "zeros"
        else:
            padding = self.padding
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode, padding_mode=padding)


class AntiAlias(torch.nn.Module):
    """Simple anti-aliasing. Based pretty much on the implementation from "Making Convolutional Networks Shift-Invariant Again"
    at https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
    """

    filter_bank = {
        1: [1.0],
        2: [1.0, 1.0],
        3: [1.0, 2.0, 1.0],
        4: [1.0, 3.0, 3.0, 1.0],
        5: [1.0, 4.0, 6.0, 4.0, 1.0],
        6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
    }

    def __init__(self, channels=3, width=5, stride=1, **kwargs):
        super().__init__()
        self.width = int(width)
        self.padding = width // 2
        self.stride = stride
        self.channels = channels

        filter_base = torch.as_tensor(self.filter_bank[self.width])
        antialias = filter_base[:, None] * filter_base[None, :]
        antialias = antialias / antialias.sum()
        self.register_buffer("antialias", antialias[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, inputs):
        return F.conv2d(inputs, self.antialias, padding=self.padding, stride=self.stride, groups=inputs.shape[1],)


augmentation_lookup = dict(
    antialias=AntiAlias,
    continuous_shift=RandomTransform,
    colorjitter=ColorJitter,
    flip=Flip,
    zoom=Zoom,
    focus=Focus,
    discrete_shift=Jitter,
    median=MedianPool2d,
    centerzoom=CenterZoom,
)
