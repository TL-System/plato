"""
Visual normalizations.


Most visual normalizations strictly follows the 'dataset_normalizations' of
https://github.com/Lightning-AI/lightning-bolts (lightning-bolts).

Sources of normalizations:
    - MNIST
        mean=(0.1307, ), std=(0.3081, ) from Plato's source code.
        mean=(0.173, ), std=(0.332, ) from the lightning-bolts.

    - CIFAR10
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])] from
            * https://github.com/leftthomas/SimCLR/blob/master/main.py
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262] from
            * https://github.com/mpatacchiola/self-supervised-relational-reasoning
            * lightning-bolts
        mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616))
            * https://www.inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/

    - CIFAR100
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            https://github.com/mpatacchiola/self-supervised-relational-reasoning

    - IMAGENET:
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] from
            * https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
            * https://github.com/PatrickHua/SimSiam/blob/main/main.py
            * lightning-bolts

    - STL10:
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            * https://github.com/mpatacchiola/self-supervised-relational-reasoning
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            * https://github.com/DarkFaceMonster/Pytorch-STL10/blob/master/model.ipynb

        mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27)
            * lightning-bolts's /models/self_supervised/cpc/transforms.py

"""

datasets_norm = {
    "MNIST": [[0.1307,], [0.3081,]],
    "FashionMNIST": [[0.1307,], [0.3081,]],
    "CIFAR10": [[0.491, 0.482, 0.447], [0.247, 0.243, 0.262]],
    "CIFAR100": [[0.491, 0.482, 0.447], [0.247, 0.243, 0.262]],
    "IMAGENET": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "STL10": [[0.4914, 0.4823, 0.4466], [0.247, 0.243, 0.261]],
}
