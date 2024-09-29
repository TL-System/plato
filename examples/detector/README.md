#Below are steps for reproduce AsyncFilter experiments:

# Installation

## Setting up your Python environment

It is recommended that [Miniforge](https://github.com/conda-forge/miniforge) is used to manage Python packages. Before using *Plato*, first install Miniforge, update your `conda` environment, and then create a new `conda` environment with Python 3.9 using the command:

```shell
conda update conda -y
conda create -n plato -y python=3.9
conda activate plato
```

where `plato` is the preferred name of your new environment.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117
```

In macOS (without GPU support), the recommended command would be:

```shell
pip install torch==1.13.1 torchvision==0.14.1
```

## Installing Plato

Additionally, we will install the current GitHub version of *Plato* as a local pip package:

```shell
pip install .
```

# Running experiments
## set up the configuration file
As we have numerous experiments in our paper, we provided some configuration files as examples. For example, for Sec. 5.2, to run asyncfilter on cifar-10 datasets:
```shell
python detector.py -c asyncfilter_cifar_2.yml
``` 
for Sec. 5.3, to run asyncilter under LIE attack on CINIC-10 dataset with the concentration factor of 0.01: 
```shell
python detector.py -c asyncfilter_cinic_3.yml
```
for Sec.5.6, to run asyncfilter under LIE attack on FashionMNIST dataset with the server staleness limit of 10: 

```shell
python detector.py -c asyncfilter_fashionmnist_6.yml
```

