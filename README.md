# Plato: A New Framework for Scalable Federated Learning Research

Welcome to *Plato*, a new software framework to facilitate scalable federated learning research.

## Installation

### Setting up your Python environment

It is recommended that [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is used to manage Python packages. Before using *Plato*, first install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), update your `conda` environment, and then create a new `conda` environment with Python 3.9 using the command:

```shell
conda update conda -y
conda create -n plato python=3.9
conda activate plato
```

where `plato` is the preferred name of your new environment.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

The CUDA version, used in the command above, can be obtained on Ubuntu Linux systems by using the command:

```shell
nvidia-smi
```

In macOS (without GPU support), the typical command would be:

```shell
conda install pytorch torchvision -c pytorch
```

### Installing Plato as a pip package

To use *Plato* as a Python framework, you only need to install it as a pip package:

```shell
pip install plato-learn
```

After *Plato* is installed, you can try to run any of the examples in `examples/`.

### Installing Plato for development with PyTorch

If you wish to modify the source code in *Plato* (rather than just using it as a framework), first clone this repository to a desired directory.

We will need to install several packages using `pip` as well:

```shell
pip install -r requirements.txt --upgrade
```

Finally, we will install the current GitHub version of *Plato* as a local pip package:

```shell
pip install .
pip install yapf mypy pylint
```

**Tip:** After the initial installation of the required Python packages, use the following command to upgrade all the installed packages at any time:

```shell
python upgrade_packages.py
```

If you use Visual Studio Code, it is possible to use `yapf` to reformat the code every time it is saved by adding the following settings to .`.vscode/settings.json`:

```
"python.formatting.provider": "yapf", 
"editor.formatOnSave": true
```

In general, the following is the recommended starting point for `.vscode/settings.json`:

```
{
	"python.linting.enabled": true,
	"python.linting.pylintEnabled": true,
	"python.formatting.provider": "yapf", 
	"editor.formatOnSave": true,
	"python.linting.pylintArgs": [
	    "--init-hook",
	    "import sys; sys.path.append('/absolute/path/to/project/home/directory')"
	],
	"workbench.editor.enablePreview": false
}
```

It goes without saying that `/absolute/path/to/project/home/directory` should be replaced with the actual path in the specific development environment.

**Tip:** When working in Visual Studio Code as your development environment, two of our colour theme favourites are called `Bluloco` (both of its light and dark variants) and `City Lights` (dark). They are both excellent and very thoughtfully designed. The `Python` extension is also required, which represents Microsoft's modern language server for Python.

### Installing YOLOv5 as a Python package

If object detection using the YOLOv5 model and any of the COCO datasets is needed, it is necessary to install YOLOv5 as a Python package first:

```shell
cd packages/yolov5
pip install .
```

## Running Plato

### Running Plato using a configuration file

To start a federated learning training workload, run [`run`](run) from the repository's root directory. For example:

```shell
./run -c configs/MNIST/fedavg_lenet5.yml
```

* `-c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

*Plato* uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

### Running Plato with MindSpore or TensorFlow

Plato is designed to support multiple deep learning frameworks, including PyTorch, TensorFlow, and MindSpore. 

**TensorFlow.** Install the `tensorflow` and `tensorflow-datasets` pip packages first:

```shell
pip install tensorflow tensorflow-datasets
./run -c configs/MNIST/fedavg_lenet5_tensorflow.yml
```

**MindSpore.** Plato currently supports the latest MindSpore release, 1.6.1. Follow the installation instructions in the [official MindSpore website](https://mindspore.cn/install/en) to install MindSpore in your conda environment. For example, on an M1 Mac, use the command:

```shell
conda install mindspore-cpu=1.6.1 -c mindspore -c conda-forge
```

To use trainers and servers based on MindSpore, assign `true` to `use_mindspore` in the `trainer` section of the configuration file. If GPU is not available when MindSpore is used, assign `true` to `cpuonly` in the `trainer` section as well. These variables are unassigned by default, and *Plato* would use PyTorch as its default framework. As examples of using MindSpore as its underlying deep learning framework, two configuration files have been provided: `configs/MNIST/fedavg_lenet5_mindspore.yml` and `configs/MNIST/mistnet_lenet5_mindspore.yml`. For example:

```shell
./run -c configs/MNIST/fedavg_lenet5_mindspore.yml
```

### Running Plato in a Docker container

Most of the codebase in *Plato* is designed to be framework-agnostic, so that it is relatively straightfoward to use *Plato* with a variety of deep learning frameworks beyond PyTorch, which is the default framwork it is using. One example of such deep learning frameworks that *Plato* currently supports is [MindSpore 1.6.1](https://www.mindspore.cn/en).

To build such a Docker image, use the provided `Dockerfile` for PyTorch and `Dockerfile_MindSpore` for MindSpore:

```shell
docker build -t plato -f Dockerfile .
```

or:

```shell
docker build -t plato -f Dockerfile_MindSpore .
```

To run the docker image that was just built, use the command:

```shell
./dockerrun.sh
```

Or if GPUs are available, use the command:

```shell
./dockerrun_gpu.sh
```

To remove all the containers after they are run, use the command:

```shell
docker rm $(docker ps -a -q)
```

To remove the `plato` Docker image, use the command:

```shell
docker rmi plato
```

On Ubuntu Linux, you may need to add `sudo` before these `docker` commands.

The provided `Dockerfile` helps to build a Docker image running Ubuntu 20.04, with a virtual environment called `plato` pre-configured to support PyTorch 1.9.0 and Python 3.9.

If MindSpore support is needed, the provided `Dockerfile_MindSpore` contains two pre-configured environments for CPU and GPU environments, respectively, called `plato_cpu` or `plato_gpu`. They support [MindSpore 1.6.1](https://github.com/mindspore-ai/mindspore) and Python 3.9.0 (which is the Python version that MindSpore 1.6.1 requires). Both Dockerfiles have GPU support enabled. Once an image is built and a Docker container is running, one can use Visual Studio Code to connect to it and start development within the container.

### Potential runtime errors

If runtime exceptions occur that prevent a federated learning session from running to completion, the potential issues could be:

* Out of CUDA memory.

  *Potential solutions:* Decrease the number of clients selected in each round (with the *client simulation mode* turned on); decrease the `max_concurrency` value in the `trainer` section in your configuration file; decrease the  `batch_size` used in the `trainer` section.
 
* The time that a client waits for the server to respond before disconnecting is too short. This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

  *Potential solutions:* Add `ping_timeout` in the `server` section in your configuration file. The default value for `ping_timeout` is 360 (seconds). 

  For example, to run a training session on [Google Colaboratory or Compute Canada](https://github.com/TL-System/plato/blob/main/docs/Running.md) with the CIFAR-10 dataset and the ResNet-18 model, and if 10 clients are selected per round, `ping_timeout` needs to be 360 when clients' local datasets are non-iid by symmetric Dirichlet distribution with the concentration of 0.01. Consider an even larger number if you run with larger models and more clients.

* Running processes have not been terminated from previous runs. 

  *Potential solutions:* Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.

### Client simulation mode

Plato supports a *client simulation mode*, in which the actual number of client processes launched equals the number of clients to be selected by the server per round, rather than the total number of clients. This supports a simulated federated learning environment, where the set of selected clients by the server will be simulated by the set of client processes actually running. For example, with a total of 10000 clients, if the server only needs to select 100 of them to train their models in each round, only 100 client processes will be launched in client simulation mode, and a client process may assume a different client ID in each round.

To turn on the client simulation mode, add `simulation: true` to the `clients` section in the configuration file.

### Server asynchronous mode

Plato supports an *asynchronous mode* for the federated learning servers. With traditional federated learning, client-side training and server-side processing proceed in a synchronous iterative fashion, where the next round of training will not commence before the current round is complete. In each round, the server would select a number of clients for training, send them the latest model, and the clients would commence training with their local data. As each client finishes its client training process, it will send its model updates to the server. The server will wait for all the clients to finish training before aggregating their model updates.

In contrast, if server asynchronous mode is activated (`server:synchronous` set to `false`), the server run its aggregation process periodically, or as soon as model updates have been received from all selected clients. The interval between periodic runs is defined in `server:periodic_interval` in the configuration. When the server runs its aggregation process, all model updates received so far will be aggregated, and new clients will be selected to replace the clients who have already sent their updates. Clients who have not sent their model updates yet will be allowed to continue their training processes. It may be the case that asynchronous mode is more efficient for cases where clients have very different training performance across the board, as faster clients may not need to wait for the slower ones (known as *stragglers* in the academic literature) to receive their freshly aggregated models from the server.

### Plotting runtime results

If the configuration file contains a `results` section, the selected performance metrics, such as accuracy, will be saved in a `.csv` file in the `results/` directory.

As `.csv` files, these results can be used however one wishes; an example Python program, called `plot.py`, plots the necessary figures and saves them as PDF files. To run this program:

```shell
python plot.py -c config.yml
```

* -c`: the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

### Running unit tests

All unit tests are in the `tests/` directory. These tests are designed to be standalone and executed separately. For example, the command `python lr_schedule_tests.py` runs the unit tests for learning rate schedules.

### Running Continuous Integration tests as GitHub actions

Continuous Integration (CI) tests have been set up for the PyTorch, TensorFlow, and MindSpore frameworks in `.github/workflows/`, and will be activated on every push and Pull Request. To run these tests manually, visit the `Actions` tab at GitHub, select the job, and then click `Run workflow`.

## Deploying Plato

### Deploying Plato servers in a production environment in the cloud

The Plato federated learning server is designed to use Socket.IO over HTTP and HTTPS, and can be easily deployed in a production server environment in the public cloud. See `/docs/Deploy.md` for more details on how the nginx web server can be used as a reverse proxy for such a deployment in production servers.

## Uninstalling Plato

Remove the `conda` environment used to run *Plato* first, and then remove the directory containing *Plato*'s git repository.

```shell
conda-env remove -n plato
rm -rf plato/
```

where `plato` (or `tensorflow` or `mindspore`) is the name of the `conda` environment that *Plato* runs in.

For more specific documentation on how Plato can be run on GPU runtime environments such as Google Colaboratory or Compute Canada, refer to `docs/Running.md`.

## Technical Support

Technical support questions should be directed to the maintainer of this software framework: Baochun Li (bli@ece.toronto.edu).
