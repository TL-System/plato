#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --account=def-baochun
#SBATCH --output=results_MAS_seed_5.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/park_env/a2c.py -c examples/park_env/a2c_MAS_seed_5.yml