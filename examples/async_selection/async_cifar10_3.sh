#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=72G
#SBATCH --output=async_CIFAR_resnet_500_20eachRound_concen10_skipBNlayer3.out


filename='./async_cifar10_resnet3.yml'
echo "The configuration filename is: $filename"
echo " " 
while read line; do
# reading each line
echo "$line"

done < $filename

python async_selection.py -c async_cifar10_resnet3.yml -b /data/ykang/plato
