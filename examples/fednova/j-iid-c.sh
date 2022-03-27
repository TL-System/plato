#!/bin/bash
#SBATCH --time=4:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-iamniudi
#SBATCH --output=o-iid-c.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python fednova.py -c configs/iid-c.yml
