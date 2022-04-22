#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --account=def-baochun
#SBATCH --output=o-f-5-n.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
./run -c configs/f-5-non.yml
