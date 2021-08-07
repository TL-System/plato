"""
The Inception v3 model for PyTorch.

Reference:

Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

"""

import torchvision


class Model():
    """The Inception v3 model."""
    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        return torchvision.models.inception_v3(aux_logits=False)
