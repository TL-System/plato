#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=out.out

python faq.py
