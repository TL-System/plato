### Personalized FL approaches with self-supervised learning

In the context of self-supervised learning (SSL), the model is trained to learn representations from unlabeled data. Thus, the model is capable of extracting generic representations. A higher performance can be achieved in subsequent tasks with the trained model as the encoder. Such a benefit of SSL is introduced into personalized FL by relying on the learning objective of SSL to train the global model. After reaching convergence, each client can download the trained global model to extract features from local samples. A high-quality personalized model, typically a linear network, is prone to be achieved under those extracted features. The code is available under `SSL/`.

- {SimCLR} [Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", Arxiv 2020, citation 374](https://arxiv.org/abs/2002.05709) - [Office code](https://github.com/google-research/simclr)


```bash
python algorithms/SSL/simclr/simclr.py -c algorithms/configs/SSL/simclr_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/simsiam/simsiam.py -c algorithms/configs/SSL/simsiam_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/smog/smog.py -c algorithms/configs/SSL/smog_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/swav/swav.py -c algorithms/configs/SSL/swav_CIFAR10_resnet18.yml -b pflExperiments
```

This mocov2 may have some problems because the loss is not decreasing.
```bash
python algorithms/SSL/moco/mocov2.py -c algorithms/configs/SSL/mocov2_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/byol/byol.py -c algorithms/configs/SSL/byol_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/fedema/fedema.py -c algorithms/configs/SSL/fedema_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/SSL/calibre/calibre.py -c algorithms/configs/SSL/calibre_CIFAR10_resnet18.yml -b pflExperiments
```
