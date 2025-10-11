# Reproducing AsyncFilter

## Setting up your Python environment

It is recommended that [Miniforge](https://github.com/conda-forge/miniforge) is used to manage Python packages. Before using *Plato*, first install Miniforge, update your `conda` environment, and then create a new `conda` environment with Python 3.9 using the command:

```bash
conda update conda -y
conda create -n plato -y python=3.9
conda activate plato
```

where `plato` is the preferred name of your new environment.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117
```

In macOS (without GPU support), the recommended command would be:

```bash
pip install torch==1.13.1 torchvision==0.14.1
```
Additionally, install scikit-learn package:

```bash
pip install scikit-learn
```
## Installing Plato

Navigate to the Plato directory and install the latest version from GitHub as a local pip package:

```bash
cd ../..
pip install .
```

# Running experiments in plato/examples/detector folder
Navigate to the examples/detector folder to start running experiments:
```bash
cd examples/detector
```

## Set up the configuration file
A variety of configuration files are provided for different experiments. Below are examples for reproducing key experiments from the paper:

### Example 1: Section 5.2 - Running AsyncFilter on CIFAR-10
#### Download the dataset

```bash
python detector.py -c asyncfilter_cifar_2.yml -d
```

#### Run the experiments
```bash
python detector.py -c asyncfilter_cifar_2.yml
```
### Example 2: Section 5.3 - Running AsyncFilter Under LIE Attack on CINIC-10 (Concentration Factor: 0.01)
#### Download the dataset

```bash
python detector.py -c asyncfilter_cinic_3.yml -d
```
#### Run the experiments
```bash
python detector.py -c asyncfilter_cinic_3.yml
```
### Example 3: Section 5.6 - Running AsyncFilter Under LIE Attack on FashionMNIST (Server Staleness Limit: 10)

#### Download the dataset

```bash
python detector.py -c asyncfilter_fashionmnist_6.yml -d
```
#### Run the experiments
```bash
python detector.py -c asyncfilter_fashionmnist_6.yml
```

### Customizing Experiments
For further experimentation, you can modify the configuration files to suit your requirements and reproduce the results.
