# Running Plato

## Running Plato on Google Colaboratory

Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb).

Under directory `plato/examples/colab/`, the notebook `colab_use_terminal.ipynb` provides step-by-step instructions on running *Plato* on Google Colaboratory, while providing the facilities to use a secure shell to login and to open Visual Studio Code. To run Plato, just use the integrated terminal in the browser.

## Running Plato on Digital Research Alliance of Canada

### Installation

SSH into a cluster on Digital Research Alliance of Canada. Here we take [Graham](https://docs.alliancecan.ca/wiki/Graham) as an example, while [Cedar](https://docs.alliancecan.ca/wiki/Cedar) and [Narval](https://docs.alliancecan.ca/wiki/Narval/en) are also available. Then clone the *Plato* repository to your own directory:

```shell
ssh <CCDB username>@graham.computecanada.ca
cd projects/def-baochun/<CCDB username>
git clone https://github.com/TL-System/plato
```

Your CCDB username can be located after signing into the [CCDB portal](https://ccdb.computecanada.ca/). Contact Baochun Li (`bli@ece.toronto.edu`) for a new account on Digital Research Alliance of Canada.

### Preparing the Python Runtime Environment

First, load version 3.9 of the Python programming language:

```shell
module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
```

Then create the directory that contains your own Python virtual environment using `virtualenv`:

```shell
virtualenv --no-download ~/.federated
```

where `~/.federated` is assumed to be the directory containing the new virtual environment just created. 

You can now activate your environment:

```shell
source ~/.federated/bin/activate
```

Due to the versioning difference for some Python packages on Digital Research Alliance of Canada, some source files may need to be patched. Use the following command to patch these files:

```shell
bash ./docs/patches/patch_cc.sh
```

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). As for January 2022, Digital Research Alliance of Canada provides GPU with CUDA version 11.2, so the command would be:

```shell
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
If it prompts `MemoryError`, use the alternative command:

```shell
pip3 install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
```

Finally, install Plato as a pip package:

```shell
pip install .
```

**Tip:** Use alias to save trouble for future running *Plato*.

```
vim ~/.bashrc
```

Then add 

```
alias plato='cd ~/projects/def-baochun/<CCDB username>/plato/; module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack; source ~/.federated/bin/activate'
```

After saving this change and exiting `vim`:

```
source ~/.bashrc
```

Next time, after you SSH into this cluster, just type `plato`:)

### Running Plato

To start a federated learning training workload with *Plato*, create a job script:

```shell
vi <job script file name>.sh
```

For exmaple:

```shell
cd ~/projects/def-baochun/<CCDB username>/plato
vi cifar_wideresnet.sh
```

Then add your configuration parameters in the job script. The following is an example:

```
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=cifar_wideresnet.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate

./run -c configs/CIFAR10/fedavg_wideresnet.yml
```

**Note:** On `cedar` and `graham`, one should remove the `--nodes=1` option.

Submit the job:

```shell
sbatch <job script file name>.sh
```

For example:

```shell
sbatch cifar_wideresnet.sh
```

To check the status of a submitted job, use the `sq` command. Refer to the [official Computer Canada documentation](https://docs/alliancecan.ca/wiki/Running_jobs#Use_sbatch_to_submit_jobs) for more details.

To monitor the output as it is generated live, use the command:

```shell
watch -n 1 tail -n 50 ./cifar_wideresnet.out
```

where `./cifar_wideresnet.out` is the output file that needs to be monitored, and the `-n` parameter for `watch` specifies the monitoring frequency in seconds (the default value is 2 seconds), and the `-n` parameter for `tail` specifies the number of lines at the end of the file to be shown. Type `Control + C` to exit the `watch` session.


**Tip:** Make sure you use different `port` numbers under `server` in different jobs' configuration files before submitting your jobs if you plan to run them at the same time. This is because they may be allocated to the same node, which is especially common when you use the `Narval` cluster. In that case, if the `port` and `address` under `server` in your configuration files of the jobs are the same, you will get `OSError: [Errno 48] error while attempting to bind on address: address already in use`.

If there is a need to start an interactive session (for debugging purposes, for example), it is also supported by Digital Research Alliance of Canada using the `salloc` command:

```shell
salloc --time=2:00:00 --gres=gpu:1 --mem=64G --account=def-baochun
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

Then you can run *Plato*:

```shell
./run -c configs/CIFAR10/fedavg_wideresnet.yml
```

After the job is done, use `exit` at the command to relinquish the job allocation.

**Note:** On Digital Research Alliance of Canada, if there are issues in the code that prevented it from running to completion, the two most possible reasons are:

If runtime exceptions occur that prevent a federated learning session from running to completion, the potential issues could be:

* Out of CUDA memory.

  *Potential solutions:* Decrease the `max_concurrency` value in the `trainer` section in your configuration file.
 
* Running processes have not been terminated from previous runs. 

  *Potential solutions:* Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.
 
* The time that a client waits for the server to respond before disconnecting is too short. This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

  *Potential solutions:* Add `ping_timeout` in the `server` section in your configuration file. The default value for `ping_timeout` is 360 (seconds). 


### Running jobs of HuggingFace

Running a job of HuggingFace requires connecting to the Internet to download the dataset and the model. However, Digital Research Alliance of Canada doesn't allow Internet connections inside sbatch/salloc. Therefore, they need to be pre-downloaded via the following steps:

1. Run the command first outside sbatch/salloc, for example, `./run -c <your configuration file>`, and use `control + C` to terminate the program right after the first client starts training. After this step, the dataset and the model should be automatically downloaded.

2. Switch to running it inside sbatch/salloc, and add `TRANSFORMERS_OFFLINE=1` before the command. The below is a sample job script:

```
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:7
#SBATCH --mem=498G
#SBATCH --account=def-baochun
#SBATCH --output=<output file>

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate

TRANSFORMERS_OFFLINE=1 ./run -c <your configuration file>
```


### Removing the Python virtual environment

To remove the environment after experiments are completed, just delete the directory:

```shell
rm -rf ~/.federated
```
