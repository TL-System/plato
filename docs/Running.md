## Running Plato on the GPU Cloud from Lambda Labs

At $1.50 an hour for a 4-GPU server with 8 VCPUs, the [GPU Cloud from Labmda Labs](https://lambdalabs.com/service/gpu-cloud) is among one of the least expensive GPU virtual machines available for lease.

(*to be completed*)

## Running Plato on Compute Canada

### Installation

SSH into `cedar.computecanada.ca` (the main server for Compute Canada) and clone the *Plato* repository to your own directory:

```shell
$ ssh <CCDB username>@cedar.computecanada.ca
$ cd projects/def-baochun/<CCDB username>
$ git clone https://github.com/baochunli/plato.git
```

Your CCDB username can be located after signing into the [CCDB portal](https://ccdb.computecanada.ca/). Contact Baochun Li (`bli@ece.toronto.edu`) for a new account on Compute Canada.

**Note:** when the command line prompts for your password for `git clone`, you should enter your [personal access token](https://github.com/settings/tokens) instead of your actual password. 

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

Install PyTorch:

```shell
$ pip install torch torchvision
```

We will need to install the `websockets` package for client-server communication:

```shell
$ pip install websockets==8.1
```

In case unit tests in the `tests` directory need to be run, `scipy` should also be installed:

```shell
$ pip install scipy
```


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
#SBATCH --time=00:15:00       # Request a job to be executed for 15 minutes
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --account=def-baochun
#SBATCH --output=cifar_wideresnet.out # The name of the output file
module load python/3.8
source ~/.federated/bin/activate
python server.py --config=configs/CIFAR10/cifar_wideresnet.conf --log=info
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
$ salloc --time=0:15:0 --ntasks=1 --cpus-per-task=2 --gres=gpu:p100l:4 --account=def-baochun
```

The job will then be queued and waiting for resources:

```
salloc: Pending job allocation 53923456
salloc: job 53923456 queued and waiting for resources
```

After the job is done, use `exit` at the command to relinquish the job allocation.

### Removing the Python virtual environment

To remove the environment after experiments are completed, just delete the directory:

```shell
$ rm -rf ~/.federated
```
