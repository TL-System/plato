# Approaches for Personalized Federated Learning

Implementations of existing approaches for personalized federated learning using the Plato framework.

--- 

## Implemented Algorithms

To better compare the performance of different personalized federated learning approaches, we implemented the following algorithms using the `pflbases` framework. The code is available under `algorithms/`.

### Baseline Algorithm

We implemented the FedAvg with finetuning, referred to as FedAvg_finetune, as the baseline algorithm for personalized federated learning (FL). The code is available under `fedavg_finetune/`.


### Classical personalized FL approaches

- {apfl} [Deng et al., "Adaptive Personalized Federated Learning", Arxiv 2020, citation 374](https://arxiv.org/pdf/2003.13461.pdf) - None

- {fedper} [Arivazhagan et al., "Federated Learning with Personalization Layers", Arxiv 2019, citation 486](https://browse.arxiv.org/pdf/1912.00818.pdf) - [Third-part Code](https://github.com/ki-ljl/FedPer)

- {ditto} [Li et.al "Ditto: Fair and robust federated learning through personalization", ICML2020, citation 419](https://proceedings.mlr.press/v139/li21h.html) - [Office code](https://github.com/litian96/ditto)

- {fedbabu} [Oh et.al "FedBABU: Toward Enhanced Representation for Federated Image Classification", ICLR 2022, citation 74](https://openreview.net/pdf?id=HuaYQfggn5u) - [Office code](https://github.com/jhoon-oh/FedBABU)

- {fedrep} [Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", ICML21, citation 289](https://arxiv.org/abs/2102.07078) - [Office code](https://github.com/lgcollins/FedRep)

- {lgfedavg} [Liang et al., "Think Locally, Act Globally: Federated Learning with Local and Global Representations", NeurIPS 2019, citation 359](https://arxiv.org/abs/2001.01523) - [Office code](https://github.com/pliang279/LG-FedAvg)

- {perfedavg} [Fallah et al., "Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach", NeurIPS 2019, citation 502](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) - [Third-part code](https://github.com/jhoon-oh/FedBABU)

- {hermes} [Li et al., "Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients", ACM MobiCom 21, citation 75](https://www.ang-li.com/assets/pdf/hermes.pdf) - None (The algorithm, first developed by Ying Chen, has been refined to integrate with the `pflbases` framework.)

### Personalized FL approaches with self-supervised learning

In the context of self-supervised learning (SSL), the model is trained to learn representations from unlabeled data. Thus, the model is capable of extracting generic representations. A higher performance can be achieved in subsequent tasks with the trained model as the encoder. Such a benefit of SSL is introduced into personalized FL by relying on the learning objective of SSL to train the global model. After reaching convergence, each client can download the trained global model to extract features from local samples. A high-quality personalized model, typically a linear network, is prone to be achieved under those extracted features. The code is available under `SSL/`.

- {SimCLR} [Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", Arxiv 2020, citation 374](https://arxiv.org/abs/2002.05709) - [Office code](https://github.com/google-research/simclr)


## Algorithms Running

1. Go to the folder `examples/personalized_fl`.
2. Install pflbases by running 

```bash
pip install .
```

3. Perfrom the algorithms by running:

```bash
python algorithms/fedavg_finetune/fedavg_finetune.py -c algorithms/configs/fedavg_finetune_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/apfl/apfl.py -c algorithms/configs/apfl_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/fedrep/fedrep.py -c algorithms/configs/fedrep_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/fedbabu/fedbabu.py -c algorithms/configs/fedbabu_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/ditto/ditto.py -c algorithms/configs/ditto_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/fedper/fedper.py -c algorithms/configs/fedper_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/lgfedavg/lgfedavg.py -c algorithms/configs/lgfedavg_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/perfedavg/perfedavg.py -c algorithms/configs/perfedavg_CIFAR10_resnet18.yml -b pflExperiments
```

```bash
python algorithms/hermes/hermes.py -c algorithms/configs/hermes_CIFAR10_resnet18.yml -b pflExperiments
```


```bash
python algorithms/SSL/simclr/simclr.py -c algorithms/configs/simclr_MNIST_lenet5.yml -b pflExperiments
python algorithms/SSL/simclr/simclr.py -c algorithms/configs/simclr_CIFAR10_resnet18.yml -b pflExperiments
```



