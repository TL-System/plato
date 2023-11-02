
## Core Components

- the _completion_ callbacks placed under `client_callbacks` merge other models into the `self.model` parameters. 
    > For example, when the whole model is A+B, you can set the global model to be A while the local model is B. Then, the completion callback will merge B into A to obtain the whole model before local update.

--- 

## Implemented Algorithms

To better compare the performance of different personalized federated learning approaches, we implemented the following algorithms.

### Baseline Algorithm

We implemented the FedAvg with finetuning, referred to as FedAvg_finetune, as the baseline algorithm for personalized federated learning (FL). The code is available under `algorithms/fedavg_finetune/`.


### Classical personalized FL approaches

- {apfl} [Deng et al., "Adaptive Personalized Federated Learning", Arxiv 2020, citation 374](https://arxiv.org/pdf/2003.13461.pdf) - None

- {fedper} [Arivazhagan et al., "Federated Learning with Personalization Layers", Arxiv 2019, citation 486](https://browse.arxiv.org/pdf/1912.00818.pdf) - [Third-party code](https://github.com/ki-ljl/FedPer)

- {ditto} [Li et.al "Ditto: Fair and robust federated learning through personalization", ICML2020, citation 419](https://proceedings.mlr.press/v139/li21h.html) - [Official code](https://github.com/litian96/ditto)

- {fedbabu} [Oh et.al "FedBABU: Toward Enhanced Representation for Federated Image Classification", ICLR 2022, citation 74](https://openreview.net/pdf?id=HuaYQfggn5u) - [Official code](https://github.com/jhoon-oh/FedBABU)

- {fedrep} [Collins et al., "Exploiting Shared Representations for Personalized Federated
Learning", ICML21, citation 289](https://arxiv.org/abs/2102.07078) - [Official code](https://github.com/lgcollins/FedRep)

- {lgfedavg} [Liang et al., "Think Locally, Act Globally: Federated Learning with Local and Global Representations", NeurIPS 2019, citation 359](https://arxiv.org/abs/2001.01523) - [Official code](https://github.com/pliang279/LG-FedAvg)

- {perfedavg} [Fallah et al., "Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach", NeurIPS 2019, citation 502](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) - [Third-party code](https://github.com/jhoon-oh/FedBABU)

- {hermes} [Li et al., "Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients", ACM MobiCom 21, citation 75](https://www.ang-li.com/assets/pdf/hermes.pdf) - None

### Personalized FL approaches with self-supervised learning

In the context of self-supervised learning (SSL), the model is trained to learn representations from unlabeled data. Thus, the model is capable of extracting generic representations. A higher performance can be achieved in subsequent tasks with the trained model as the encoder. Such a benefit of SSL is introduced into personalized FL by relying on the learning objective of SSL to train the global model. After reaching convergence, each client can download the trained global model to extract features from local samples. A high-quality personalized model, typically a linear network, is prone to be achieved under those extracted features. The code is available under `SSL/`.

- {SimCLR} [Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", Arxiv 2020, citation 374](https://arxiv.org/abs/2002.05709) - [Office code](https://github.com/google-research/simclr)


2. Perform the algorithms by running:

```bash
python algorithms/fedavg_finetune/fedavg_finetune.py -c algorithms/configs/fedavg_finetune_CIFAR10_resnet18.yml -b pflExperiments
```

---

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

---

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



## Hyper-parameters

All hyper-parameters should be placed under the `algorithm` block of the configuration file. 

### For `fedavg_personalized`
- local_layer_names: This is a list in which each item is a string presenting the parameter name. When you utilize the `fedavg_personalized.py` as the algorithm, the `local_layer_names` is required to be set under the `algorithm` block of the configuration file. Then, only the parameters contained in the `local_layer_names` will be the global model to be exchanged between the server and clients. Thus, server aggregation will be performed only on these parameters. If this hyper-parameter is not set, all parameters of the defined model will be used by default. For example, 
    ```yaml
    algorithm:
        local_layer_names:
            - conv1
            - bn1
            - layer1
            - layer2
            - layer3
            - layer4
    ```

- local_layer_names: This is a list in which each item is a string presenting the parameter name. Once you set the `local_layer_names`, the client receives a portion of the model from the server. To embrace all parameters (i.e., the whole model) during the local update, you should set `local_layer_names` to indicate which parts parameters will be loaded from the local side to merge with the received ones. This is more like: the whole model is A+B. The client receives A from the server. Then, the client loads B from the local side to merge with A. For example, 
    ```yaml
    algorithm:
        local_layer_names:
            - linear
    ```

### For Personalization

All hyper-parameters related to personalization should be placed under the `personalization` sub-block of the `algorithm` block. 

- model_name: A string to indicate the personalized model name. This is not mandatory as if it is omitted, it will be assumed to be the same as the global model. Default: `model_name` under the `trainer` block.  For example, 
    ```yaml
    algorithm:
        personalization:
            # the personalized model name
            # this can be omitted
            model_name: resnet_18
    ```

- participating_clients_ratio: A float to show the proportion of clients participating in the federated training process. The value ranges from 0.0 to 1.0 while 1.0 means that all clients will participant in training. Default: 1.0. For example, 
    ```yaml
    algorithm:
        personalization:
            # the ratio of clients participanting in training
            participating_clients_ratio: 0.6
    ```


