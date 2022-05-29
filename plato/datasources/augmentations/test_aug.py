""" The implementation of data augmentation for the test phase 
    of contrastive learning.
    
    The test phase of the contrastive learning is to measure 
    the quality of representation. Thus, the clustering
    methods can be used to first learn from the extracted 
    representation of the trainset. Then, the learned clusters 
    are used to obtain accuracy from the testset.
"""

import torchvision.transforms as T
from PIL import Image


class TestTransform():

    def __init__(self, image_size, normalize):

        transform_functions = [
            T.Resize(image_size),  # 224 -> 256 
            T.ToTensor(),
        ]
        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):
        return self.transform(x)
