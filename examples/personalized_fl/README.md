# Algorithms for Personalized Federated Learning

Implementations of existing approaches for personalized federated learning using the Plato framework.

## Two Basic Learning Modes
`pflbases` contains two basic learning modes, i.e., performing personalization in each round or the next round of the final round. The indicator is `do_personalization_per_round` (see hyper-parameters below).

When `do_personalization_per_round: true`, the personalized model will be 1. Loaded; 2. Obtained; 3. Saved, in each round. Otherwise, the personalized model will only be processed in the next round of the final round. 

We believe that these two modes can cover all personalized FL algorithms. For example, 
- APFL, Ditto, LG-FedAvg, FedPer, FedRep, and FedBABU follows `do_personalization_per_round: true`. 
- FedAvg with finetuning (fedavg_finetune) and PerFeAavg follows `do_personalization_per_round: false`.

--- 
## Baseline Algorithm

We implemented the FedAvg with finetuning, referred to as FedAvg_finetune, as the baseline algorithm for personalized federated learning. The code is available under `fedavg_finetune/`.

## Implemented Algorithms

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


## Hyper-parameters

All hyper-parameters of `pflbases` should be placed under the `algorithm` block of the configuration file. 

### For `fedavg_partial`
- global_modules_name: This is a list in which each item is a string presenting the parameter name. When you utilize the `fedavg_partial.py` as the algorithm, the `global_modules_name` is required to be set under the `algorithm` block of the configuration file. Then, only the parameters contained in the `global_modules_name` will be the global model to be exchanged between the server and clients. Thus, server aggregation will be performed only on these parameters. If this hyper-parameter is not set, all parameters of the defined model will be used by default. For example, 
```yaml
algorithm:
    global_modules_name:
        - conv1
        - bn1
        - layer1
        - layer2
        - layer3
        - layer4
```

- completion_modules_name: This is a list in which each item is a string presenting the parameter name. Once you set the `global_modules_name`, the client receives a portion of the model from the server. To embrace all parameters (i.e., the whole model) during the local update, you should set `completion_modules_name` to indicate which parts parameters will be loaded from the local side to merge with the received ones. This is more like: the whole model is A+B. The client receives A from the server. Then, the client loads B from the local side to merge with A. For example, 
```yaml
algorithm:
    completion_modules_name:
        - linear
```

### For Personalization

All hyper-parameters related to personalization should be placed under the `personalization` sub-block of the `algorithm` block. 

- model_name: A string to indicate the personalized model name. For example, 
```yaml
algorithm:
    personalization:

        model_name: resnet_18
```

- participating_clients_ratio: A float to show the proporation of clients participating in the federated training process. The value ranges from 0.0 to 1.0 while 1.0 means that all clients will participant in training. Default: 1.0. For example, 
```yaml
algorithm:
    personalization:
        # the ratio of clients participanting in training
        participating_clients_ratio: 0.6
```

- do_personalization_per_round: A boolean to indicate whether to perform personalization per round. Default: True. For example, 
```yaml
algorithm:
    personalization:

        # whether the personalized model will be
        # obtained in each round
        do_personalization_per_round: true
```
