## Plato: A New Framework for Federated Learning Research

Welcome to *Plato*, a new software framework to facilitate scalable federated learning research.

### Installing Plato with PyTorch

To install *Plato*, first clone this repository to the desired directory.

The *Plato* developers recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages. Before using *Plato*, first install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), update your `conda` environment, and then create a new `conda` environment with Python 3.8 using the command:

```shell
$ conda update conda
$ conda create -n federated python=3.8
$ conda activate federated
```

where `federated` is the the preferred name of your new environment.

Update any packages, if necessary by typing `y` to proceed.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
$ conda install pytorch torchvision cudatoolkit=11.1 -c pytorch
```

The CUDA version, used in the command above, can be obtained on Ubuntu Linux systems by using the command:

```shell
nvidia-smi
```

In macOS (without GPU support), the typical command would be:

```shell
$ conda install pytorch torchvision -c pytorch
```

We will need to install several packages using `pip` as well:

```shell
$ pip install -r requirements.txt
```

If you use Visual Studio Code, it is possible to use `yapf` to reformat the code every time it is saved by adding the following settings to .`.vscode/settings.json`:

```
"python.formatting.provider": "yapf", 
"editor.formatOnSave": true
```

In general, the following is the recommended starting point for `.vscode/settings.json`:

```
"python.linting.enabled": true,
"python.linting.pylintEnabled": true,
"python.formatting.provider": "yapf", 
"editor.formatOnSave": true,
"python.linting.pylintArgs": [
    "--init-hook",
    "import sys; sys.path.append('/absolute/path/to/project/home/directory')"
],
"workbench.editor.enablePreview": false
```

It goes without saying that `/absolute/path/to/project/home/directory` should be replaced with the actual path in the specific development environment.

**Tip:** When working in Visual Studio Code as the development environment, one of the project developer's colour theme favourites is called `Bluloco`, both of its light and dark variants are excellent and very thoughtfully designed. The `Pylance` extension is also strongly recommended, which represents Microsoft's modern language server for Python.

### Running Plato in a Docker container

Most of the codebase in *Plato* is designed to be framework-agnostic, so that it is relatively straightfoward to use *Plato* with a variety of deep learning frameworks beyond PyTorch, which is the default framwork it is using. One example of such deep learning frameworks that *Plato* currently supports is [MindSpore](https://www.mindspore.cn). Due to the wide variety of tricks that need to be followed correctly for running *Plato* without Docker, it is strongly recommended to run Plato in a Docker container, on either a CPU-only or a GPU-enabled server.

To build such a Docker image, use the provided `Dockerfile` for PyTorch and `Dockerfile_MindSpore` for MindSpore:

```shell
docker build -t plato -f Dockerfile .
docker build -t plato_mindspore -f Dockerfile_MindSpore .
```

To run any of the docker images that was just built, use the command:

```shell
docker run -it --net=host -v /dev/shm:/dev/shm plato bash
```

Or if GPUs are available, use the command:

```shell
docker run -it --net=host -v /dev/shm:/dev/shm --gpus all plato bash
```

To remove all the containers after they are run, use the command:

```shell
docker rm $(docker ps -a -q)
```

To remove the `plato` Docker image, use the command:

```shell
docker rmi plato plato_mindspore
```

On Ubuntu Linux, you may need to add `sudo` before these `docker` commands.

The provided `Dockerfile` helps to build a Docker image running Ubuntu 20.04, with a virtual environment called `federated` pre-configured to support PyTorch 1.8.1 and Python 3.8. If MindSpore support is needed, the provided `Dockerfile_MindSpore` contains a pre-configured environment, also called `federated`, that supports [MindSpore 1.1.1](https://github.com/mindspore-ai/mindspore) and Python 3.7.5 (which is the Python version that MindSpore requires). Both Dockerfiles have GPU support enabled. Once an image is built and  a container is running, one can use Visual Studio Code to connect to it and start development within the container.

### Installing Plato with MindSpore

Though we provided a `Dockerfile` for building a Docker container that supports MindSpore 1.1, in rare cases it may still be necessary to install Plato with MindSpore in a GPU server running Ubuntu Linux 18.04 (which MindSpore requires). Similar to a PyTorch installation, we need to first create a new environment with Python 3.7.5 (which MindSpore 1.1 requires), and then install the required packages:

```shell
conda create -n mindspore python=3.7.5
pip install -r requirements.txt
```

We should now install MindSpore 1.1 with the following command:
```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.1/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-1.1.1-cp37-cp37m-linux_x86_64.whl
```

MindSpore may need additional packages that need to be installed if they do not exist:

```shell
sudo apt-get install libssl-dev
sudo apt-get install build-essential
```

If CuDNN has not yet been installed, it needs to be installed with the following commands:

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install libcudnn8=8.0.5.39-1+cuda10.1
```

To check the current CuDNN version, the following commands are helpful:

```shell
function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }
check libcudnn
```

To check if MindSpore is correctly installed on the GPU server, try to `import mindspore` with a Python interpreter.

Finally, to use trainers and servers based on MindSpore, assign `true` to `use_mindspore` in the `trainer` section of the configuration file. This variable is unassigned by default, and *Plato* would use PyTorch as its default framework.

### Running Plato

To start a federated learning training workload, run [`run`](run) from the repository's root directory. For example:

```shell
./run --config=configs/MNIST/fedavg_lenet5.yml
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.yml` in the project's home directory.
* `--log` (`-l`): the level of logging information to be written to the console. Possible values are `critical`, `error`, `warn`, `info`, and `debug`, and the default is `info`.

*Plato* uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

*Plato* uses `wandb` to produce and collect logs in the cloud. If this is not needed, run the command `wandb offline` before running *Plato*.

If there are issues in the code that prevented it from running to completion, there could be running processes from previous runs. Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.

### Installing YOLOv5 as a Python package

If object detection using the YOLOv5 model and any of the COCO datasets is needed, it is required to install YOLOv5 as a Python package first:

```shell
cd packages/yolov5
pip install .
```

### Plotting Runtime Results

If the configuration file contains a `results` section, the selected performance metrics, such as accuracy, will be saved in a `.csv` file in the `results/` directory. By default, the `results/` directory is under the path to the used configuration file, but it can be easily changed by modifying `Config.result_dir` in [`config.py`](config.py).

As `.csv` files, these results can be used however one wishes; an example Python program, called `plot.py`, plots the necessary figures and saves them as PDF files. To run this program:

```shell
python plot.py --config=config.yml
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

### Running Unit Tests

All unit tests are in the `tests/` directory. These tests are designed to be standalone and executed separately. For example, the command `python lr_schedule_tests.py` runs the unit tests for learning rate schedules.

### Uninstalling Plato

Remove the `conda` environment used to run *Plato* first, and then remove the directory containing *Plato*'s git repository.

```shell
conda-env remove -n federated
rm -rf plato/
```

where `federated` (or `mindspore`) is the name of the `conda` environment that *Plato* runs in.

For more specific documentation on how Plato can be run on GPU cluster environments such as Lambda Labs' GPU cloud or Compute Canada, refer to `docs/Running.md`.

### Technical support

Technical support questions should be directed to the maintainer of this software framework: Baochun Li (bli@ece.toronto.edu).
