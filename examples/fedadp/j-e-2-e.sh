#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=126G
#SBATCH --account=def-baochun
#SBATCH --output=o-e-2-e.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python fedadp.py -c configs/e-2-ext.yml
