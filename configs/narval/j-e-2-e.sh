#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --account=def-baochun
#SBATCH --output=o-e-2-e.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
./run -c configs/e-2-ext.yml
