#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results_fisher.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/park_env/a2c.py -c examples/park_env/a2c_fisher_aggregate.yml

#To run, connect to Graham, follow the Running.md file under docs, move this file to the main plato directory and run from there
#Remember to copy over data folder as Graham does not allow external downloads
#Tip: make the output file seed specific