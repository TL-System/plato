#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results_fisher.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source .federated/bin/activate
python a2c.py -c a2c_fisher_aggregate.yml