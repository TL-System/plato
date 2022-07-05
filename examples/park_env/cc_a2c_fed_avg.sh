#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=def-baochun
#SBATCH --output=results_fed_avg_seed_5.out

module load gcc/9.3.0 arrow cuda/11 python/3.9 scipy-stack
source ~/.federated/bin/activate
python examples/park_env/a2c.py -c examples/park_env/a2c_fedavg_seed_5.yml

#To run, connect to Graham, follow the Running.md file under docs, move this file to the main plato directory and run from there
#Remember to copy over data folder as Graham does not allow external downloads
#Tip: make the output file seed specific
#Make sure that before submitting another batch that it runs the yml file you want, otherwise
#if you make changes to your yml file it will run that LATEST version of that yml file
#change port numbers everytime