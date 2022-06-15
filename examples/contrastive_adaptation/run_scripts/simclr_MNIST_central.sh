#!/bin/bash 
#SBATCH --time=5:00:00 
#SBATCH --cpus-per-task=21 
#SBATCH --gres=gpu:1 
#SBATCH --mem=100G 
#SBATCH --output=./slurm_loggings/simclr_MNIST_central.out 
 
/home/sijia/envs/miniconda3/envs/INFOCOM23/bin/python ./examples/contrastive_adaptation/simclr/simclr.py -c ./examples/contrastive_adaptation/configs/simclr_MNIST_central.yml -b /data/sijia/INFOCOM23/experiments 
