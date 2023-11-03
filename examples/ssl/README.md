### Personalized FL approaches with self-supervised learning

In the context of self-supervised learning (SSL), the model is trained to learn representations from unlabeled data. Thus, the model is capable of extracting generic representations. A higher performance can be achieved in subsequent tasks with the trained model as the encoder. Such a benefit of SSL is introduced into personalized FL by relying on the learning objective of SSL to train the global model. After reaching convergence, each client can download the trained global model to extract features from local samples. A high-quality personalized model, typically a linear network, is prone to be achieved under those extracted features. The code is available under `examples/ssl/`. And under `algorithms/` of the folder, the following algorithms are implemented:

- {SimCLR} [Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020, citation 12967](https://arxiv.org/abs/2002.05709) - [Office code](https://github.com/google-research/simclr)
    ```bash
    python algorithms/simclr/simclr.py -c algorithms/configs/simclr_CIFAR10_resnet18.yml -b pflExperiments
    ```

- {SimSiam} [Chen et al., "Exploring Simple Siamese Representation Learning", CVPR 2021, citation 2847](https://arxiv.org/abs/2011.10566) - [Office code](https://github.com/facebookresearch/simsiam)

    ```bash
    python algorithms/simsiam/simsiam.py -c algorithms/configs/simsiam_CIFAR10_resnet18.yml -b pflExperiments
    ```

- {SMoG} [Pang et al., "Unsupervised Visual Representation Learning by Synchronous Momentum Grouping", ECCV 2022, citation 109](https://arxiv.org/abs/2207.06167) - [Office code](https://bopang1996.github.io/files/SMoG_code.zip)
    ```bash
    python algorithms/smog/smog.py -c algorithms/configs/smog_CIFAR10_resnet18.yml -b pflExperiments
    ```

- {SwaV} [Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", NeurIPS 2022, citation 2791](https://arxiv.org/abs/2011.10566) - [Office code](https://github.com/facebookresearch/simsiam)

```bash
python algorithms/swav/swav.py -c algorithms/configs/swav_CIFAR10_resnet18.yml -b pflExperiments
```

- {MoCoV2} [Chen et al., "Improved Baselines with Momentum Contrastive Learning", Arxiv 2022, citation 2603](https://arxiv.org/abs/2011.10566) - [Office code](https://github.com/facebookresearch/moco)

This mocov2 may have some problems because the loss is not decreasing.
```bash
python algorithms/moco/mocov2.py -c algorithms/configs/mocov2_CIFAR10_resnet18.yml -b pflExperiments
```

- {BYOL} [Grill et al., "Bootstrap your own latent: A new approach to self-supervised Learning", NeurIPS 2020, citation 4686](https://arxiv.org/abs/2006.07733) - [Office code](https://github.com/lucidrains/byol-pytorch)
```bash
python algorithms/byol/byol.py -c algorithms/configs/byol_CIFAR10_resnet18.yml -b pflExperiments
```

- {FedEMA} [Grill et al., "Divergence-aware Federated Self-Supervised Learning", ICLR 2022, citation 154](https://arxiv.org/pdf/2204.04385.pdf) - [Office code](https://github.com/lucidrains/byol-pytorch)
```bash
python algorithms/fedema/fedema.py -c algorithms/configs/fedema_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/calibre/calibre.py -c algorithms/configs/calibre_CIFAR10_resnet18.yml -b pflExperiments
```
