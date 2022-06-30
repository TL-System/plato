#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results_2.out

module load python
source ~/.federated/bin/activate
python a2c.py -c a2c_fisher_aggregate.yml