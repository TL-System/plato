"""
The commonly used normalization values for different datasets.

"""

datasets_norm = {
    "MNIST": [(0.1307, ), (0.3081, )],
    "CIFAR10": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "IMAGENET": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
}
