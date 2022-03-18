#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --account=def-iamniudi
#SBATCH --output=o-non-c.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python fedprox.py -c configs/non-c.yml
