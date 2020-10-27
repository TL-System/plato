## Running Plato on Compute Canada

### Prerequisites

SSH into Cedar and clone the *Plato* repository to your own directory:

```shell
$ ssh <CCDB username>@cedar.computecanada.ca
$ cd projects/def-iamniudi/<CCDB username>
$ git clone https://github.com/baochunli/plato.git
```

**Note:** when the command line prompts for your password for `git clone`, you should enter your [personal access token](https://github.com/settings/tokens) instead of your actual password. 

Load python version 3.8:

```shell
$ module load python/3.8
```

Create directory for your own environment:

```shell
$ virtualenv --no-download ~/ENV
```

Activate your environment:

```shell
$ source ~/ENV/bin/activate
```

Install PyTorch:

```shell
$ pip install torch torchvision
```

### Simulation

To start a simulation, create a job script:

```shell
$ vi <job script file name>.sh
```

For exmaple, 

```shell
$ vi mnist_cnn.sh
```

Then type your configurations in the job script. The following is an example:

```
#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --account=def-iamniudi
#SBATCH --output=mnist_cnn.out
module load python/3.8
source ~/ENV/bin/activate
python run.py --config=configs/MNIST/mnist.conf --log=INFO
```

**Note:** the GPU resource required in the above example, is a special group of GPU nodes on Cedar. You may only request these nodes as whole nodes, therefore you must specify --gres=gpu:p100l:4. P100L GPU jobs up to 28 days can be run on Cedar.
You may use any type of GPU available on Compute Canada, but I find under most cases using this P100L GPU requires less waiting time for jobs requiring long time.

Submit the job:

```shell
$ sbatch <job script file name>.sh
```

For exmaple, 

```shell
$ sbatch mnist_cnn.sh
```


