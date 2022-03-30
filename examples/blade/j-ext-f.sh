#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-iamniudi
#SBATCH --output=o-ext-f.out
module load python/3.9
source ~/.federated/bin/activate
python blade.py -c configs/ext-f.yml
