import numpy as np
from torch.utils.data import Dataset


class CustomDictDataset(Dataset):
    """Custom dataset from a dictionary with support of transforms."""
    def __init__(self, dictionary, transform=None):
        self.xs = dictionary['x']
        self.ys = dictionary['y']
        self.transform = transform

    def __getitem__(self, index):
        x = self.xs[index]
        if self.transform:
            x = self.transform(x)
        y = self.ys[index]
        return x, y

    def __len__(self):
        return len(self.xs)


class ReshapeListTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, img):
        return np.array(img, dtype=np.float32).reshape(self.new_shape)