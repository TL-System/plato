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
$ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
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

### Installing Plato with MindSpore

Most of the codebase in *Plato* is designed to be framework-agnostic, so that it is relatively straightfoward to use *Plato* with a variety of deep learning frameworks beyond PyTorch, which is the default framwork it is using. One example of such deep learning frameworks is [MindSpore](https://www.mindspore.cn).

Though we provided a `Dockerfile` in `docker/` for building a Docker container that supports MindSpore 1.1, it may still be necessary to install Plato with MindSpore in a GPU server running Ubuntu Linux 18.04 (which MindSpore requires). Similar to a PyTorch installation, we need to first create a new environment with Python 3.7.5 (which MindSpore 1.1 requires), and then install the required packages:

```shell
conda create -n mindspore python=3.7.5
conda install matplotlib pylint yapf scipy
pip install websockets requests
```

We should now install MindSpore 1.1 with the following command:
```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/gpu/ubuntu_x86/cuda-10.1/mindspore_gpu-1.1.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com
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

To start a federated learning training workload, run [`server.py`](server.py) from the repository's root directory. For example:

```shell
cp configs/MNIST/fedavg_lenet5.yml config.yml
python server.py --config=config.yml --log=info
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.yml` in the project's home directory.
* `--log` (`-l`): the level of logging information to be written to the console. Possible values are `critical`, `error`, `warn`, `info`, and `debug`, and the default is `info`.

*Plato* uses the YAML format for its configuration files to manage the runtime configuration parameters. Example configuration files have been provided in the `configs` directory.

### Plotting Runtime Results

If the configuration file contains a `results` section, the selected performance metrics, such as accuracy, will be saved in a `.csv` file in the `results/` directory. By default, the `results/` directory is under the path to the used configuration file, but it can be easily changed by modifying `Config.result_dir` in [`config.py`](config.py).

As `.csv` files, these results can be used however one wishes; an example Python program, called `plot.py`, plots the necessary figures and saves them as PDF files. To run this program:

```shell
python plot.py --config=config.yml
```

* `--config` (`-c`): the path to the configuration file to be used. The default is `config.yml` in the project's home directory.

### Running Unit Tests

All unit tests are in the `tests/` directory. These tests are designed to be standalone and executed separately. For example, the command `python lr_schedule_tests.py` runs the unit tests for learning rate schedules.

### Building a Docker container for running Plato

Sometimes it may be beneficial to run Plato in a Docker container. To build such such a Docker container, use the provided `Dockerfile` in `docker/`:

```shell
cd docker; docker build -t plato .
```

To run the docker image that was just built, use the command:

```shell
docker run -it --net=host plato
```

To remove all the containers after they are run, use the command:

```shell
docker rm $(docker ps -a -q)
```

To remove the `plato` Docker image, use the command:

```shell
docker rmi plato
```

The provided `Dockerfile` helps to build a Docker container image running Ubuntu 20.04, with two virtual environments pre-installed: the one called `federated` supports PyTorch and Python 3.8, and one called `mindspore` supports [MindSpore 1.1](https://github.com/mindspore-ai/mindspore) and Python 3.7.5 (which is the Python version that MindSpore requires). Once the container is built and running, one can use Visual Studio Code to connect to it and start development within the container.

### Maintaining external repositories as subrepos

The *Plato* framework uses external git repositories by cloning them as *subrepos* in the `subrepos/` directory. Most collaborators or users do not need to worry about maintaining these subrepos. In order to update these subrepos, the [git subrepo](https://github.com/ingydotnet/git-subrepo) command needs to be installed. To install it, use the following commands:

```shell
git clone https://github.com/ingydotnet/git-subrepo /path/to/git-subrepo
echo 'source /path/to/git-subrepo/.rc' >> ~/.bashrc
source ~/.bashrc
```

After installing `git subrepo`, one can clone an external git repository using the following commands from the *top-level* project directory:

```shell
git subrepo clone https://github.com/ultralytics/yolov5.git subrepos/yolov5
```

It is more typical that one needs to pull the latest updates from the external git repository that has already been cloned as a subrepo:

```shell
git subrepo pull subrepos/yolov5
```

Where `yolov5` is the external repo to be cloned in our example. Before cloning a new subrepo, one needs to make sure that the working directory has neither unstaged nor staged changes.

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
