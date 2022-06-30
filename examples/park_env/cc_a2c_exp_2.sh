#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results.out

module load python
source ~/.federated/bin/activate
python examples/park_env/a2c.py -c examples/park_env/a2c_FashionMNIST_lenet5_2.yml