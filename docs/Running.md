## Running Plato on Google Colaboratory

### Option 1

Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb).

Click `File` on the menu (upper left of the page), select `Upload Notebook`, and upload `plato_colab.ipynb`, which is under the `plato/examples/` directory.

### Option 2

Click [this link](https://colab.research.google.com/drive/1boDurcQF5X9jq25-DsKDTus3h50NBn8h?usp=sharing).

## Running Plato on Compute Canada

### Installation

SSH into a cluster on Compute Canada. Here we take [Béluga](https://docs.computecanada.ca/wiki/Béluga/en) as an example, while [Graham](https://docs.computecanada.ca/wiki/Graham) and [Cedar]((https://docs.computecanada.ca/wiki/Cedar)) are also available. Then clone the *Plato* repository to your own directory:

```shell
$ ssh <CCDB username>@beluga.computecanada.ca
$ cd projects/def-baochun/<CCDB username>
$ git clone https://github.com/TL-System/plato.git
```

Your CCDB username can be located after signing into the [CCDB portal](https://ccdb.computecanada.ca/). Contact Baochun Li (`bli@ece.toronto.edu`) for a new account on Compute Canada.

Change the permissions on `plato` directory:

```shell
$chmod 777 -R plato/
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

Currently there is no feasiable way to install the `datasets` package with `virtualenvs` (Compute Canada asks users to not use Conda.) So we cannot run *Plato* with HuggingFace datasets on Compute Canada right now. If you want to do experiments with HuggingFace datasets, you could do it on Google Colaboratory.

To bypass installing the `datasets` package, please comment out the following 2 lines of code related to `huggingface` in `plato/datasources/huggingface.py`:

```
# from datasets import load_dataset
# self.dataset = load_dataset(dataset_name, dataset_config)
```

Also, modify one line in `setup.py`. Change

```
def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()
```

to

```
def get_requirements():
    with open("./docs/cc_requirements.txt") as f:
        return f.read().splitlines()
```

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). Currently Compute Canada provides GPU with CUDA version 11.2, so the command would be:

```shell
$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

To double-check the CUDA version used in the command above, start an interactive session and use the following command:

```shell
nvidia-smi
```

Finally, install Plato as a pip package:

```shell
$ pip install .
```

**Tip:** Use alias to save trouble for future running *Plato*.

```
$ vim ~/.bashrc
```

Then add 

```
alias plato='cd ~/projects/def-baochun/<CCDB username>/plato/; module load python/3.8; source ~/.federated/bin/activate'
```

After saving this change and exiting `vim`, 

```
$ source ~/.bashrc
```

Next time, after you SSH into this cluster, just type `plato`:)



### Running Plato

To start a federated learning training workload with *Plato*, create a job script:

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
#SBATCH --time=20:00:00  # Request a job to be executed for 20 hours
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=191000M
#SBATCH --nodes=1
#SBATCH --account=def-baochun
#SBATCH --output=cifar_wideresnet.out # The name of the output file
module load python/3.8
source ~/.federated/bin/activate
./run --config=configs/CIFAR10/fedavg_wideresnet.yml --log=info
```

**Note:** The above example requests a type of GPU on Compute Canada's `Béluga` cluster that requires a very short waiting time (as for July 2021, but things may change.)

You may use any type of [GPUs available on Compute Canada](https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm).

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
$ salloc --time=0:20:0 --ntasks=1 --cpus-per-task=4 --gres=gpu:4 --account=def-baochun --mem=191000M
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
$ module load python/3.8
$ virtualenv --no-download ~/.federated
$ ./run --config=configs/CIFAR10/fedavg_wideresnet.yml
```

After the job is done, use `exit` at the command to relinquish the job allocation.

**Note:** On Compute Canada, if there are issues in the code that prevented it from running to completion, the two most possible reasons are:

If runtime exceptions occur that prevent a federated learning session from running to completion, the potential issues could be:

* Out of CUDA memory.

  *Potential solutions:* Decrease the number of clients selected in each round (with the *client simulation mode* turned on); decrease the `max_concurrency` value in the `trainer` section in your configuration file; decrease the  `batch_size` used in the `trainer` section.
 
* The time that a client waits for the server to respond before disconnecting is too short. This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

  *Potential solutions:* Add `ping_timeout` in the `server` section in your configuration file. The default value for `ping_timeout` is 360 (seconds). 

  For example, to run a training session with the CIFAR-10 dataset and the ResNet-18 model, and if 10 clients are selected per round, `ping_timeout` needs to be 360 when clients' local datasets are non-iid by symmetric Dirichlet distribution with the concentration of 0.01. Consider an even larger number if you run with larger models and more clients.

* Running processes have not been terminated from previous runs. 

  *Potential solutions:* Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.

### Removing the Python virtual environment

To remove the environment after experiments are completed, just delete the directory:

```shell
$ rm -rf ~/.federated
```
