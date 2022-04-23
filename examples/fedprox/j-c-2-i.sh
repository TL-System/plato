#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --account=def-baochun
#SBATCH --output=o-c-2-i.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python fedprox.py -c configs/c-2-iid.yml
