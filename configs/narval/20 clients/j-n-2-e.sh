#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --account=def-baochun
#SBATCH --output=o-n-2-e.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
./run -c configs/n-2-ext.yml
