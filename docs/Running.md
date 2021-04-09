## Running Plato on the GPU Cloud from Lambda Labs

At $1.50 an hour for a 4-GPU server with 8 VCPUs, the [GPU Cloud from Labmda Labs](https://lambdalabs.com/service/gpu-cloud) is among one of the least expensive GPU virtual machines available for lease.

(*to be completed*)

## Running Plato on Google Colaboratory

Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb).

Click `File` on the menu (upper left of the page), select `Upload Notebook`, and upload `plato_colab.ipynb`, which is under `plato/examples/` directory.

## Running Plato on Compute Canada

### Installation

SSH into `cedar.computecanada.ca` (the main server for Compute Canada) and clone the *Plato* repository to your own directory:

```shell
$ ssh <CCDB username>@cedar.computecanada.ca
$ cd projects/def-baochun/<CCDB username>
$ git clone https://github.com/TL-System/plato.git
```

Your CCDB username can be located after signing into the [CCDB portal](https://ccdb.computecanada.ca/). Contact Baochun Li (`bli@ece.toronto.edu`) for a new account on Compute Canada.

**Note:** when the command line prompts for your password for `git clone`, you should enter your [personal access token](https://github.com/settings/tokens) instead of your actual password. 

Change the permissions on `plato` directory:"

```shell
$chmod 777 -R plato/"
```

### Preparing the Python Runtime Environment

First, you need to load version 3.8 of the Python programming language:

```shell
$ module load python/3.8
```

Then, you need to create the directory that contains your own Python virtual environment using `virtualenv`:

```shell
$ virtualenv --no-download ~/.federated
```

where `~/.federated` is assumed to be the directory containing the new virtual environment just created. 

You can now activate your environment:

```shell
$ source ~/.federated/bin/activate
```

Install required packages using `pip`:

```shell
$ pip install -r docs/cc_requirements.txt --no-index
```

The `--no-index` option tells `pip` to not install from PyPI, but only from locally-available packages, i.e. the Compute Canada wheels.
Whenever a Compute Canada wheel is available for a given package, it is strongly recommended to use it by way of the `--no-index` option. Compared to using packages from PyPI, wheels that have been compiled by Compute Canada staff can prevent issues with missing or conflicting dependencies, and were optimised for its clusters hardware and libraries. 

**Note:** I am still working on finding an available way to install the `datasets` package with `virtualenvs` (Compute Canada asks users to not use Conda.) So we cannot run Plato with HuggingFace datasets right now. Please comment out 2 lines of code related to `huggingface` in `datasources/registry.py` before your experiments for now.


### Running Plato

To start a federated learning training workload with Plato, create a job script:

```shell
$ vi <job script file name>.sh
```

For exmaple:

```shell
$ cd ~/projects/def-baochun/<CCDB username>/plato
$ vi cifar_wideresnet.sh
```

Then add your configuration parameters in the job script. The following is an example:

```
#!/bin/bash
#SBATCH --time=15:00:00       # Request a job to be executed for 15 hours
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --account=def-baochun
#SBATCH --output=cifar_wideresnet.out # The name of the output file
module load python/3.8
source ~/.federated/bin/activate
./run --config=configs/CIFAR10/fedavg_wideresnet.yml --log=info
```

**Note:** the GPU resources requested in this example is a special group of GPU nodes on Compute Canada's `cedar` cluster. You may only request these nodes as whole nodes, therefore you must specify `--gres=gpu:p100l:4`. NVIDIA P100L GPU jobs up to 28 days can be run on the `cedar` cluster.

You may use any type of GPU available on Compute Canada, but in most cases using the NVIDIA P100L GPU requires shorter waiting times, especially for jobs requiring long running times.

Submit the job:

```shell
$ sbatch <job script file name>.sh
```

For example:

```shell
$ sbatch cifar_wideresnet.sh
```

To check the status of a submitted job, use the `sq` command. Refer to the [official Computer Canada documentation](https://docs.computecanada.ca/wiki/Running_jobs#Use_sbatch_to_submit_jobs) for more details.

To monitor the output as it is generated live, use the command:

```shell
$ watch -n 1 tail -n 50 ./cifar_wideresnet.out
```

where `./cifar_wideresnet.out` is the output file that needs to be monitored, and the `-n` parameter for `watch` specifies the monitoring frequency in seconds (the default value is 2 seconds), and the `-n` parameter for `tail` specifies the number of lines at the end of the file to be shown. Type `Control + C` to exit the `watch` session.

If there is a need to start an interactive session (for debugging purposes, for example), it is also supported by Compute Canada using the `salloc` command:

```shell
$ salloc --time=0:15:0 --ntasks=1 --cpus-per-task=4 --gres=gpu:p100l:4 --mem=32G --account=def-baochun
```

The job will then be queued and waiting for resources:

```
salloc: Pending job allocation 53923456
salloc: job 53923456 queued and waiting for resources
```

As soon as your job gets resources, you get the following prompts:

```
salloc: job 53923456 has been allocated resources
salloc: Granted job allocation 53923456
```

Then you can run Plato:

```shell
$ module load python/3.8
$ virtualenv --no-download ~/.federated
$ ./run --config=configs/CIFAR10/fedavg_wideresnet.yml
```


After the job is done, use `exit` at the command to relinquish the job allocation.

### Removing the Python virtual environment

To remove the environment after experiments are completed, just delete the directory:

```shell
$ rm -rf ~/.federated
```
