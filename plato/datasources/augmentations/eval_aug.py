""" The implementation of data augmentation for the eval phase 
    of contrastive learning.
    
    In general, the evaluation phase is implemented by the 
        linear evaluation. This is a typical transform.

"""

from torchvision import transforms
from PIL import Image


class EvalTransform():

    def __init__(self, image_size, train, normalize):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),  # 224 -> 256 
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)
