## Installation

SSH into a cluster on Digital Research Alliance of Canada. Here we take [Narval](https://docs.alliancecan.ca/wiki/Narval) as an example, while [Rorqual](https://docs.alliancecan.ca/wiki/Rorqual/en) is also available.

```bash
ssh <CCDB username>@narval.computecanada.ca
cd projects/def-baochun/<CCDB username>
```

!!! note "Note"
    You could also use `/scratch/<CCDB username>` to store temporary files using next command.
    ```bash
    cd /scratch/<CCDB username>
    ```

Then clone the *Plato* repository to your own directory:
```
git clone https://github.com/TL-System/plato
```

Your CCDB username can be located after signing into the [CCDB portal](https://ccdb.computecanada.ca/). Contact Baochun Li (`bli@ece.toronto.edu`) for a new account on Digital Research Alliance of Canada.

## Preparing the Python Runtime Environment

First, load version 3.12 of the Python programming language:

To discover the versions of Python available:

```bash
module avail python
```

Load version 3.12 of the Python programming language:

```bash
module load python/3.12
```

You can then create your own Python virtual environment (for example, one called .federated):
```bash
virtualenv --no-download ~/.federated # creating your own virtual environment
source ~/.federated/bin/activate
```

Then install uv if it's not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Now you can run Plato with `uv run`:

```bash
uv run --active plato.py -c configs/MNIST/fedavg_lenet5.yml
```

In case you wish to exit your Python virtual environment, run the command:
```bash
deactivate
```

!!! note "Note"
    Use alias to save trouble for future running *Plato*.

    ```
    vim ~/.bashrc
    ```

    Then add

    ```
    alias plato='cd ~/projects/def-baochun/<CCDB username>/plato/; module load python/3.12; source ~/.federated/bin/activate'
    ```

    After saving this change and exiting `vim`:

    ```
    source ~/.bashrc
    ```

    Next time, after you SSH into this cluster, just type `plato`:)

## Running Plato

To start a federated learning training workload with *Plato*, create a job script:

```bash
vi <job script file name>.sh
```

For exmaple:

```bash
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

module load python/3.12
source ~/.federated/bin/activate

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv run --active plato.py -c configs/CIFAR10/fedavg_wideresnet.yml
```

Submit the job:

```bash
sbatch <job script file name>.sh
```

For example:

```bash
sbatch cifar_wideresnet.sh
```

To check the status of a submitted job, use the `sq` command. Refer to the [official Computer Canada documentation](https://docs.alliancecan.ca/wiki/Running_jobs#Use_squeue_or_sq_to_list_jobs) for more details.

To monitor the output as it is generated live, use the command:

```bash
watch -n 1 tail -n 50 ./cifar_wideresnet.out
```

where `./cifar_wideresnet.out` is the output file that needs to be monitored, and the `-n` parameter for `watch` specifies the monitoring frequency in seconds (the default value is 2 seconds), and the `-n` parameter for `tail` specifies the number of lines at the end of the file to be shown. Type `Control + C` to exit the `watch` session.

!!! note "Tip"
    Make sure you use different `port` numbers under `server` in different jobs' configuration files before submitting your jobs if you plan to run them at the same time. This is because they may be allocated to the same node, which is especially common when you use the `Narval` cluster. In that case, if the `port` and `address` under `server` in your configuration files of the jobs are the same, you will get `OSError: [Errno 48] error while attempting to bind on address: address already in use`.

If there is a need to start an interactive session (for debugging purposes, for example), it is also supported by Digital Research Alliance of Canada using the `salloc` command:

```bash
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

```bash
uv run --active plato.py -c configs/CIFAR10/fedavg_wideresnet.yml
```

After the job is done, use `exit` at the command to relinquish the job allocation.

!!! note "Note"
    On the Digital Research Alliance of Canada, if there are issues in the code that prevent it from running to completion, the potential issues could be:

    !!! tip "Out of CUDA memory."
        Potential solutions: Decrease the `max_concurrency` value in the `trainer` section in your configuration file.

    !!! tip "Running processes have not been terminated from previous runs."
        Potential solutions: Use the command `pkill python` to terminate them so that there will not be CUDA errors in the upcoming run.

    !!! tip "The time that a client waits for the server to respond before disconnecting is too short."
        This could happen when training with large neural network models. If you get an `AssertionError` saying that there are not enough launched clients for the server to select, this could be the reason. But make sure you first check if it is due to the *out of CUDA memory* error.

        Potential solutions: Add `ping_timeout` in the `server` section in your configuration file. The default value for `ping_timeout` is 360 (seconds).


### Running jobs of HuggingFace

Running a job of HuggingFace requires connecting to the Internet to download the dataset and the model. However, Digital Research Alliance of Canada doesn't allow Internet connections inside sbatch/salloc. Therefore, they need to be pre-downloaded via the following steps:

1. Run the command first outside sbatch/salloc, for example, `uv run --active plato.py -c <your configuration file>`, and use `control + C` to terminate the program right after the first client starts training. After this step, the dataset and the model should be automatically downloaded.

2. Switch to running it inside sbatch/salloc, and add `TRANSFORMERS_OFFLINE=1` before the command. The below is a sample job script:

```
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=498G
#SBATCH --account=def-baochun
#SBATCH --output=test.out

# Limit OpenBLAS threads
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load python/3.12
source ~/.federated/bin/activate

# Running the test.sh under examples/async/fedbuff
TRANSFORMERS_OFFLINE=1 uv run --active fedbuff.py -c fedbuff_cifar10.yml
```


### Removing the Python virtual environment

To remove the environment after experiments are completed, just delete the directory:

```bash
rm -rf ~/.federated
```
