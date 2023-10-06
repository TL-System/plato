# Approaches for Personalized Federated Learning

Implementations of existing approaches for personalized federated learning using the Plato framework.

--- 
## Baseline Algorithm

We implemented the FedAvg with finetuning, referred to as FedAvg_finetune, as the baseline algorithm for personalized federated learning. The code is available under `fedavg_finetune/`.

## Implemented Algorithms

- {apfl} [Deng et.al, "Adaptive Personalized Federated Learning"](https://arxiv.org/pdf/2003.13461.pdf) - None

- {ditto} [Li et.al "Ditto: Fair and robust federated learning through personalization"](https://proceedings.mlr.press/v139/li21h.html) - [Office code](https://github.com/litian96/ditto)

- {fedbabu} [Oh et.al "FedBABU: Toward Enhanced Representation for Federated Image Classification"](https://openreview.net/pdf?id=HuaYQfggn5u) - [Office code](https://github.com/jhoon-oh/FedBABU)

- {fedrep} [Collins et.al, "Exploiting Shared Representations for Personalized Federated
Learning"](https://arxiv.org/abs/2102.07078) - [Office code](https://github.com/lgcollins/FedRep)

- {lgfedavg} [Liang et.al, "Think Locally, Act Globally: Federated Learning with Local and Global Representations"](https://arxiv.org/abs/2001.01523) - [Office code](https://github.com/pliang279/LG-FedAvg)

- {perfedavg} [Fallah et.al, "Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach"](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) - [Third-part code](https://github.com/jhoon-oh/FedBABU)


## Algorithms Running

1. Go to the folder `examples/pfl`.
2. Install pflbases by running 

```bash
pip install .
```

3. Perfrom the algorithm, such as fedrep by running

```bash
python algorithms/fedrep/fedrep.py -c algorithms/configs/fedrep_CIFAR10_resnet18.yml -b pflExperiments
```